"""
scripts/run_envB_v2_bc_formal_validation.py
EnvB_v2 BC formal validation — 20 seeds per dataset.

Frozen config (per ENVB_V2_PRE_FORMAL_DECISION.md):
  Environment: EnvB_v2
  small-wide:   50k,  families A+B+C, delay=0.25, seed=425
  large-narrow: 200k, family A only,  delay=0.10, seed=411
  Training:     BC × 20 seeds (0-19), same hyperparameters as EnvA_v2 BC
  Eval:         greedy, 50 episodes on EnvB_v2 START→GOAL

Features:
  - skip-completed (reads existing CSV, skips already-done rows)
  - crash-safe append (opens CSV in append mode after first row)
  - dataset existence / metadata check; regenerates if stale or missing
  - checkpoints saved to artifacts/training_validation_v2/envB_v2_bc_checkpoints/

Outputs:
  artifacts/training_validation_v2/envB_v2_bc_formal_summary.csv
  artifacts/training_validation_v2/envB_v2_bc_checkpoints/<dataset>_seed<N>.pt
  docs/ENVB_V2_BC_FORMAL_RUNLOG.md
"""

import sys, os, csv, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from envs.gridworld_envs import EnvB_v2, N_ACTIONS
from scripts.build_envB_v2_datasets import (
    build_all, WIDE_CONFIG, NARROW_CONFIG,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR       = os.path.join(PROJECT_ROOT, "artifacts", "training_validation_v2")
CKPT_DIR      = os.path.join(OUT_DIR, "envB_v2_bc_checkpoints")
DATASET_DIR   = os.path.join(PROJECT_ROOT, "artifacts", "envB_v2_datasets")
DOCS_DIR      = os.path.join(PROJECT_ROOT, "docs")
SUMMARY_PATH  = os.path.join(OUT_DIR, "envB_v2_bc_formal_summary.csv")
RUNLOG_PATH   = os.path.join(DOCS_DIR, "ENVB_V2_BC_FORMAL_RUNLOG.md")

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Frozen experiment config ───────────────────────────────────────────────────

FORMAL_SEEDS  = list(range(20))    # 0..19
EVAL_EPISODES = 50
ENV_NAME      = "EnvB_v2"
ALGO          = "bc"

# BC hyperparameters — identical to EnvA_v2 BC formal experiment
BC_CFG = {
    "hidden_dims":  [256, 256],
    "num_updates":  5000,
    "batch_size":   256,
    "lr":           3e-4,
    "weight_decay": 1e-4,
}

# Dataset frozen metadata for staleness check
DATASET_SPECS = {
    "envB_v2_small_wide": {
        "n_transitions":    WIDE_CONFIG["n_trans"],
        "seed":             WIDE_CONFIG["seed"],
        "delay_prob":       WIDE_CONFIG["delay_prob"],
        "source_families":  ["A", "B", "C"],
    },
    "envB_v2_large_narrow_A": {
        "n_transitions":    NARROW_CONFIG["n_trans"],
        "seed":             NARROW_CONFIG["seed"],
        "delay_prob":       NARROW_CONFIG["delay_prob"],
        "source_families":  ["A"],
    },
}

SUMMARY_COLUMNS = [
    "env_name", "dataset_name", "algorithm", "seed",
    "avg_return", "success_rate", "avg_episode_len",
    "final_train_loss", "checkpoint_path", "status",
]

# ── Network ────────────────────────────────────────────────────────────────────

_G      = EnvB_v2.grid_size   # 20
OBS_DIM = _G * _G              # 400


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        L = []; d = in_dim
        for h in hidden:
            L += [nn.Linear(d, h), nn.ReLU()]; d = h
        L.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)

# ── Observation encoding ───────────────────────────────────────────────────────

def encode_obs(raw):
    """(N,2) int32 → (N, OBS_DIM) float32 one-hot."""
    n = len(raw)
    out = np.zeros((n, OBS_DIM), dtype=np.float32)
    out[np.arange(n), raw[:, 0] * _G + raw[:, 1]] = 1.0
    return out


def encode_single(r, c):
    v = torch.zeros(OBS_DIM, dtype=torch.float32)
    v[int(r) * _G + int(c)] = 1.0
    return v

# ── Dataset loader ─────────────────────────────────────────────────────────────

def check_dataset(ds_name):
    """Return True if .npz exists and n_transitions metadata matches spec."""
    path = os.path.join(DATASET_DIR, f"{ds_name}.npz")
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path)
        stored_n = int(d["n_transitions"][0])
        expected_n = DATASET_SPECS[ds_name]["n_transitions"]
        return stored_n == expected_n
    except Exception:
        return False


def load_dataset_b2(ds_name):
    """Load EnvB_v2 .npz and return one-hot encoded float32 arrays."""
    path = os.path.join(DATASET_DIR, f"{ds_name}.npz")
    d = np.load(path)
    obs   = encode_obs(d["observations"].astype(np.int32))
    nobs  = encode_obs(d["next_observations"].astype(np.int32))
    acts  = d["actions"].astype(np.int64)
    rews  = d["rewards"].astype(np.float32)
    terms = d["terminals"].astype(np.float32)
    return obs, acts, rews, nobs, terms

# ── Training loop ──────────────────────────────────────────────────────────────

def train_bc(obs, acts, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n     = len(obs)
    model = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    opt   = optim.Adam(model.parameters(),
                       lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    obs_t  = torch.from_numpy(obs)
    acts_t = torch.from_numpy(acts)
    final_loss = float("nan")
    for _ in range(cfg["num_updates"]):
        idx    = np.random.randint(0, n, size=cfg["batch_size"])
        loss   = nn.functional.cross_entropy(model(obs_t[idx]), acts_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        final_loss = loss.item()
    return model, final_loss

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, n_episodes=EVAL_EPISODES):
    env = EnvB_v2()
    returns, succs, lens = [], [], []
    model.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs_raw, _ = env.reset()
            ret = 0.0; ep_len = 0; done = False; success = False
            while not done:
                logits = model(encode_single(*obs_raw).unsqueeze(0))
                a = int(logits.argmax(1).item())
                obs_raw, r, term, trunc, _ = env.step(a)
                ret += r; ep_len += 1; done = term or trunc
                if term: success = True
            returns.append(ret); succs.append(float(success)); lens.append(ep_len)
    model.train()
    return float(np.mean(returns)), float(np.mean(succs)), float(np.mean(lens))

# ── Skip-completed helper ──────────────────────────────────────────────────────

def load_completed(summary_path):
    """Return set of (dataset_name, seed) already in summary CSV."""
    done = set()
    if not os.path.exists(summary_path):
        return done
    try:
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") in ("completed", "ok"):
                    done.add((row["dataset_name"], int(row["seed"])))
    except Exception:
        pass
    return done


def append_row(summary_path, row, write_header=False):
    """Append one result row to CSV (crash-safe append mode)."""
    mode = "a" if os.path.exists(summary_path) and not write_header else "w"
    with open(summary_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header or mode == "w":
            w.writeheader()
        w.writerow(row)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("EnvB_v2 BC Formal Validation — 20 seeds")
    print("=" * 68)

    # ── 1. Dataset check / regenerate ─────────────────────────────────────
    print("\n── 1. Dataset check ──")
    regen_needed = False
    for ds_name in ["envB_v2_small_wide", "envB_v2_large_narrow_A"]:
        ok = check_dataset(ds_name)
        print(f"  {ds_name}: {'OK' if ok else 'MISSING/STALE — will regenerate'}")
        if not ok:
            regen_needed = True

    if regen_needed:
        print("  Regenerating datasets...")
        build_all(verbose=True)
        for ds_name in ["envB_v2_small_wide", "envB_v2_large_narrow_A"]:
            assert check_dataset(ds_name), f"Regeneration failed for {ds_name}"
        print("  Datasets ready.")

    # ── 2. Load datasets ───────────────────────────────────────────────────
    print("\n── 2. Loading datasets ──")
    datasets = {}
    for ds_name, label in [
        ("envB_v2_small_wide",      "small-wide"),
        ("envB_v2_large_narrow_A",  "large-narrow-A"),
    ]:
        obs, acts, rews, nobs, terms = load_dataset_b2(ds_name)
        spec = DATASET_SPECS[ds_name]
        print(f"  {ds_name}: {len(obs)} transitions  "
              f"(expected {spec['n_transitions']})")
        assert len(obs) == spec["n_transitions"], "Dataset size mismatch"
        datasets[ds_name] = (obs, acts)

    # ── 3. Load completed seeds to support resume ──────────────────────────
    completed = load_completed(SUMMARY_PATH)
    print(f"\n── 3. Resume check: {len(completed)} seeds already done ──")

    write_header = not os.path.exists(SUMMARY_PATH)
    if write_header:
        # Create file with header only
        with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS).writeheader()

    total_runs   = len(FORMAL_SEEDS) * 2  # 20 seeds × 2 datasets
    completed_n  = len(completed)
    remaining_n  = total_runs - completed_n
    print(f"  Total runs: {total_runs}  Remaining: {remaining_n}")

    # ── 4. Training loop ───────────────────────────────────────────────────
    print(f"\n── 4. Training (BC × {len(FORMAL_SEEDS)} seeds × 2 datasets) ──")
    run_count = 0

    for ds_name, (obs, acts) in datasets.items():
        print(f"\n  Dataset: {ds_name}")
        for seed in FORMAL_SEEDS:
            if (ds_name, seed) in completed:
                print(f"    seed={seed:2d}  SKIP (already done)")
                continue

            ckpt_name = f"{ds_name}_seed{seed:02d}.pt"
            ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

            # Check if checkpoint already exists but not in CSV (partial crash)
            if os.path.exists(ckpt_path):
                try:
                    state = torch.load(ckpt_path, weights_only=True)
                    model = MLP(OBS_DIM, N_ACTIONS, BC_CFG["hidden_dims"])
                    model.load_state_dict(state)
                    avg_ret, succ_rt, avg_len = evaluate(model)
                    final_loss = float("nan")  # not recoverable from ckpt alone
                    print(f"    seed={seed:2d}  RESTORED from checkpoint  "
                          f"ret={avg_ret:.4f}  succ={succ_rt:.3f}")
                    row = {
                        "env_name": ENV_NAME, "dataset_name": ds_name,
                        "algorithm": ALGO, "seed": seed,
                        "avg_return": f"{avg_ret:.4f}",
                        "success_rate": f"{succ_rt:.4f}",
                        "avg_episode_len": f"{avg_len:.2f}",
                        "final_train_loss": "restored",
                        "checkpoint_path": ckpt_path,
                        "status": "completed",
                    }
                    append_row(SUMMARY_PATH, row)
                    run_count += 1
                    continue
                except Exception:
                    pass  # corrupt checkpoint, retrain

            # Train
            print(f"    seed={seed:2d}  training...", end=" ", flush=True)
            model, final_loss = train_bc(obs, acts, BC_CFG, seed)
            avg_ret, succ_rt, avg_len = evaluate(model)
            print(f"loss={final_loss:.4f}  ret={avg_ret:.4f}  succ={succ_rt:.3f}")

            # Save checkpoint
            torch.save(model.state_dict(), ckpt_path)

            row = {
                "env_name": ENV_NAME, "dataset_name": ds_name,
                "algorithm": ALGO, "seed": seed,
                "avg_return": f"{avg_ret:.4f}",
                "success_rate": f"{succ_rt:.4f}",
                "avg_episode_len": f"{avg_len:.2f}",
                "final_train_loss": f"{final_loss:.6f}",
                "checkpoint_path": ckpt_path,
                "status": "completed",
            }
            append_row(SUMMARY_PATH, row)
            run_count += 1

    # ── 5. Analysis ────────────────────────────────────────────────────────
    print(f"\n── 5. Analysis ──")
    results = {"envB_v2_small_wide": [], "envB_v2_large_narrow_A": []}
    with open(SUMMARY_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["dataset_name"]
            if ds in results and row["status"] == "completed":
                try:
                    results[ds].append(float(row["avg_return"]))
                except ValueError:
                    pass

    sw_rets = np.array(results["envB_v2_small_wide"])
    ln_rets = np.array(results["envB_v2_large_narrow_A"])

    def ci95(arr):
        if len(arr) < 2: return float("nan"), float("nan")
        se = arr.std(ddof=1) / math.sqrt(len(arr))
        return arr.mean() - 1.96*se, arr.mean() + 1.96*se

    sw_mean, sw_std = float(sw_rets.mean()), float(sw_rets.std(ddof=1))
    ln_mean, ln_std = float(ln_rets.mean()), float(ln_rets.std(ddof=1))
    gap = sw_mean - ln_mean
    sw_lo, sw_hi = ci95(sw_rets)
    ln_lo, ln_hi = ci95(ln_rets)

    print(f"  small-wide     n={len(sw_rets)}  mean={sw_mean:.4f}  std={sw_std:.4f}  "
          f"95%CI=[{sw_lo:.4f},{sw_hi:.4f}]")
    print(f"  large-narrow-A n={len(ln_rets)}  mean={ln_mean:.4f}  std={ln_std:.4f}  "
          f"95%CI=[{ln_lo:.4f},{ln_hi:.4f}]")
    print(f"  gap (sw - ln) = {gap:.4f}")

    sw_wins = int(np.sum(sw_rets > ln_rets)) if (len(sw_rets)==len(ln_rets)) else -1
    directional = (gap > 0)
    ci_no_overlap = (sw_lo > ln_hi) if (not math.isnan(sw_lo) and not math.isnan(ln_hi)) else False

    # ── 6. Write RUNLOG ────────────────────────────────────────────────────
    recommend_iql = directional  # proceed to IQL if BC shows positive gap

    lines = [
        "# ENVB_V2_BC_FORMAL_RUNLOG.md",
        "# EnvB_v2 BC Formal Validation — 20 Seeds",
        "",
        "> Date: 2026-04-06",
        f"> Runs completed this session: {run_count}",
        f"> Total rows in summary CSV: {len(sw_rets) + len(ln_rets)}",
        "",
        "---",
        "",
        "## 1. Experiment Config",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Environment | EnvB_v2 |",
        f"| Algorithm | BC (Behavior Cloning) |",
        f"| Seeds | 0–19 (20 per dataset) |",
        f"| Eval episodes | {EVAL_EPISODES} |",
        f"| MLP hidden | [256, 256] |",
        f"| Updates | {BC_CFG['num_updates']} |",
        f"| Batch size | {BC_CFG['batch_size']} |",
        f"| LR | {BC_CFG['lr']} |",
        f"| small-wide  | 50k, A+B+C, delay=0.25, seed=425 |",
        f"| large-narrow-A | 200k, family A, delay=0.10, seed=411 |",
        f"| SA coverage gap | 0.2407 − 0.0852 = 0.1556 |",
        "",
        "## 2. Results",
        "",
        "### small-wide (50k, A+B+C)",
        "",
        f"n = {len(sw_rets)}",
        f"mean return  = {sw_mean:.4f}",
        f"std          = {sw_std:.4f}",
        f"95% CI       = [{sw_lo:.4f}, {sw_hi:.4f}]",
        f"min / max    = {float(sw_rets.min()):.4f} / {float(sw_rets.max()):.4f}",
        "",
        "### large-narrow-A (200k, family A)",
        "",
        f"n = {len(ln_rets)}",
        f"mean return  = {ln_mean:.4f}",
        f"std          = {ln_std:.4f}",
        f"95% CI       = [{ln_lo:.4f}, {ln_hi:.4f}]",
        f"min / max    = {float(ln_rets.min()):.4f} / {float(ln_rets.max()):.4f}",
        "",
        "### Comparison",
        "",
        f"Gap (sw_mean − ln_mean) = {gap:.4f}",
    ]

    if len(sw_rets) == len(ln_rets):
        lines.append(f"Seed-paired: wide > narrow in {sw_wins}/{len(sw_rets)} seeds")
    lines += [
        f"95% CI overlap: {'NO (non-overlapping)' if ci_no_overlap else 'YES (overlapping)'}",
        "",
        "## 3. Directional Assessment",
        "",
    ]

    if directional and ci_no_overlap:
        lines += [
            "**STRONG directional support for wide > narrow.**",
            "",
            f"- Mean gap = {gap:.4f} (positive)",
            "- 95% CI intervals do not overlap",
            "- EnvB_v2 BC results are consistent with the coverage hypothesis",
        ]
    elif directional:
        lines += [
            "**DIRECTIONAL support for wide > narrow (CIs overlap).**",
            "",
            f"- Mean gap = {gap:.4f} (positive)",
            "- 95% CI intervals overlap — the effect is directional but not decisive at 20 seeds",
            "- Consistent with coverage hypothesis; CIs overlapping at n=20 is expected",
            "  given the moderate effect size in small-grid environments",
        ]
    else:
        lines += [
            "**NO directional support: wide NOT > narrow at mean level.**",
            "",
            f"- Mean gap = {gap:.4f} (non-positive)",
            "- This does not refute the coverage hypothesis — BC on a single-goal task",
            "  may be topology-insensitive (as noted in v4 fairness audit)",
        ]

    lines += [
        "",
        "## 4. Proceed to IQL Formal 20 Seeds?",
        "",
    ]
    if recommend_iql:
        lines += [
            "**YES — proceed to IQL formal 20 seeds.**",
            "",
            "Rationale:",
            f"- BC shows positive gap ({gap:.4f}), consistent with coverage hypothesis",
            "- IQL smoke passed (stable at 5,000 updates, success_rate=1.000)",
            "- IQL is more capable than BC — should amplify the coverage contrast",
        ]
    else:
        lines += [
            "**CONDITIONAL — consider before proceeding to IQL.**",
            "",
            "BC did not show positive gap. Before running IQL 20 seeds:",
            "- Check whether the gap is near-zero (topology-insensitive) or clearly negative",
            "- If clearly negative: investigate design confound or accept EnvB_v2 as",
            "  learning-insensitive and use it only as a structural validation environment",
            "- If near-zero: IQL may still show positive gap — proceed cautiously",
        ]

    lines += [
        "",
        "## 5. Per-Seed Returns",
        "",
        "| seed | small-wide | large-narrow-A |",
        "|------|------------|----------------|",
    ]
    for i in range(min(len(sw_rets), len(ln_rets))):
        lines.append(f"| {i:2d} | {sw_rets[i]:.4f} | {ln_rets[i]:.4f} |")
    for i in range(len(sw_rets), len(ln_rets)):
        lines.append(f"| {i:2d} | — | {ln_rets[i]:.4f} |")
    for i in range(len(ln_rets), len(sw_rets)):
        lines.append(f"| {i:2d} | {sw_rets[i]:.4f} | — |")

    with open(RUNLOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Runlog: {RUNLOG_PATH}")
    print(f"  Summary CSV: {SUMMARY_PATH}")

    print()
    print("=" * 68)
    print(f"small-wide:     mean={sw_mean:.4f}  std={sw_std:.4f}")
    print(f"large-narrow-A: mean={ln_mean:.4f}  std={ln_std:.4f}")
    print(f"Gap:            {gap:.4f}")
    print(f"CI non-overlap: {ci_no_overlap}")
    print(f"Directional:    {'YES (wide > narrow)' if directional else 'NO'}")
    print(f"Proceed to IQL: {'YES' if recommend_iql else 'CONDITIONAL'}")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    sys.exit(main())
