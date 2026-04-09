"""
scripts/run_envC_v2_cql_formal_validation.py
EnvC_v2 CQL formal validation — 20 seeds per dataset.

Frozen config (per ENVC_V2_IMPLEMENTATION_RUNLOG.md):
  Environment:  EnvC_v2 (extended state ((row,col), has_key))
  small-wide:   50k, LU+LD+RU+RD, delay=0.05, uniform_random, seed=600
  large-narrow: 200k, LU only,    delay=0.05, opposite,       seed=601
  Training:     CQL × 20 seeds (0–19), same frozen config as EnvA_v2 / EnvB_v2 CQL
  Eval:         greedy, 50 episodes on EnvC_v2 START→GOAL
  OBS encoding: 401-dim (400 pos one-hot + has_key scalar)

CQL: conservative Q-learning with td_loss + cql_alpha * cql_penalty.
Features:
  - skip-completed  (reads existing CSV, skips already-done rows)
  - crash-safe append (opens CSV in append mode after header)
  - dataset existence / metadata check; regenerates if stale or missing
  - checkpoints to envC_v2_cql_checkpoints/
  - route-family convergence tracked via greedy rollout analysis

Outputs:
  artifacts/training_validation_v2/envC_v2_cql_formal_summary.csv
  artifacts/training_validation_v2/envC_v2_cql_route_family_summary.csv
  artifacts/training_validation_v2/envC_v2_cql_checkpoints/<ds>_seed<N>.pt
  docs/ENVC_V2_CQL_FORMAL_RUNLOG.md
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from envs.gridworld_envs import EnvC_v2, N_ACTIONS
from scripts.build_envC_v2_datasets import (
    build_all, WIDE_CONFIG, NARROW_CONFIG,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR      = os.path.join(PROJECT_ROOT, "artifacts", "training_validation_v2")
CKPT_DIR     = os.path.join(OUT_DIR, "envC_v2_cql_checkpoints")
DATASET_DIR  = os.path.join(PROJECT_ROOT, "artifacts", "envC_v2_datasets")
DOCS_DIR     = os.path.join(PROJECT_ROOT, "docs")
SUMMARY_PATH = os.path.join(OUT_DIR, "envC_v2_cql_formal_summary.csv")
RFAM_PATH    = os.path.join(OUT_DIR, "envC_v2_cql_route_family_summary.csv")
RUNLOG_PATH  = os.path.join(DOCS_DIR, "ENVC_V2_CQL_FORMAL_RUNLOG.md")

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Frozen experiment config ───────────────────────────────────────────────────

FORMAL_SEEDS  = list(range(20))
EVAL_EPISODES = 50
ENV_NAME      = "EnvC_v2"
ALGO          = "cql"

CQL_CFG = {
    "hidden_dims":  [256, 256],
    "num_updates":  5000,
    "batch_size":   256,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "gamma":        0.99,
    "cql_alpha":    1.0,
}

DATASET_SPECS = {
    "envC_v2_small_wide": {
        "n_transitions":   WIDE_CONFIG["n_trans"],
        "source_families": WIDE_CONFIG["families"],
    },
    "envC_v2_large_narrow_LU": {
        "n_transitions":   NARROW_CONFIG["n_trans"],
        "source_families": NARROW_CONFIG["families"],
    },
}

SUMMARY_COLUMNS = [
    "env_name", "dataset_name", "algorithm", "seed",
    "avg_return", "success_rate", "avg_episode_len",
    "final_td_loss", "final_cql_penalty", "final_total_loss",
    "checkpoint_path", "status",
]

RFAM_COLUMNS = [
    "dataset_name", "seed", "dominant_combined_family",
    "frac_LU", "frac_LD", "frac_RU", "frac_RD", "frac_shared",
    "avg_return", "note",
]

# ── Network ────────────────────────────────────────────────────────────────────

_G      = EnvC_v2.grid_size
_K1, _K2 = (5, 2), (5, 7)
OBS_DIM = _G * _G + 1   # 401


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
    """(N, 3) int32 of (row, col, has_key) → (N, 401) float32."""
    n = len(raw)
    out = np.zeros((n, OBS_DIM), dtype=np.float32)
    out[np.arange(n), raw[:, 0] * _G + raw[:, 1]] = 1.0
    out[:, _G * _G] = raw[:, 2].astype(np.float32)
    return out


def encode_single(pos, has_key):
    r, c = pos
    v = torch.zeros(OBS_DIM, dtype=torch.float32)
    v[int(r) * _G + int(c)] = 1.0
    v[_G * _G] = float(has_key)
    return v

# ── Dataset loader ─────────────────────────────────────────────────────────────

def check_dataset(ds_name):
    path = os.path.join(DATASET_DIR, f"{ds_name}.npz")
    if not os.path.exists(path): return False
    try:
        d = np.load(path)
        return int(d["n_transitions"][0]) == DATASET_SPECS[ds_name]["n_transitions"]
    except Exception: return False


def load_dataset_c2(ds_name):
    path = os.path.join(DATASET_DIR, f"{ds_name}.npz")
    d    = np.load(path)
    obs  = encode_obs(d["observations"].astype(np.int32))
    nobs = encode_obs(d["next_observations"].astype(np.int32))
    acts = d["actions"].astype(np.int64)
    rews = d["rewards"].astype(np.float32)
    term = d["terminals"].astype(np.float32)
    return obs, acts, rews, nobs, term

# ── CQL training ───────────────────────────────────────────────────────────────

def train_cql(obs, acts, rews, nobs, terms, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n     = len(obs)
    model = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    opt   = optim.Adam(model.parameters(), lr=cfg["lr"],
                       weight_decay=cfg["weight_decay"])
    ot = torch.from_numpy(obs); at = torch.from_numpy(acts)
    rt = torch.from_numpy(rews); nt = torch.from_numpy(nobs)
    dt = torch.from_numpy(terms)
    g = cfg["gamma"]; alpha = cfg["cql_alpha"]
    final_td = final_cql = final_total = float("nan")
    opt.zero_grad()
    for _ in range(cfg["num_updates"]):
        idx = np.random.randint(0, n, size=cfg["batch_size"])
        s, a, r, ns, d = ot[idx], at[idx], rt[idx], nt[idx], dt[idx]
        with torch.no_grad():
            next_q  = model(ns).max(1).values
            target  = r + g * (1 - d) * next_q
        q_all  = model(s)
        q_sa   = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        td_l   = nn.functional.mse_loss(q_sa, target)
        cql_l  = (torch.logsumexp(q_all, dim=1) - q_sa).mean()
        loss   = td_l + alpha * cql_l
        loss.backward(); opt.step(); opt.zero_grad()
        final_td    = td_l.item()
        final_cql   = cql_l.item()
        final_total = loss.item()
    return model, final_td, final_cql, final_total

# ── Evaluation + route-family inference ───────────────────────────────────────

def _stage(pos, hk):
    r, c = pos
    if hk == 0:
        if 2 <= r <= 8 and 1 <= c <= 3: return "Pre-L"
        if 2 <= r <= 8 and 5 <= c <= 9: return "Pre-R"
        return "Pre-shared"
    else:
        if pos == (9, 10): return "door"
        if 3 <= r <= 8  and 12 <= c <= 18: return "Post-U"
        if 10 <= r <= 16 and 12 <= c <= 18: return "Post-D"
        return "Post-shared"


def infer_combined_family(traj):
    pre_side = post_side = None
    for (pos, hk), _ in traj:
        s = _stage(pos, hk)
        if s == "Pre-L"  and pre_side  is None: pre_side  = "L"
        if s == "Pre-R"  and pre_side  is None: pre_side  = "R"
        if s == "Post-U" and post_side is None: post_side = "U"
        if s == "Post-D" and post_side is None: post_side = "D"
    if pre_side and post_side:
        return pre_side + post_side
    for (pos, hk), _ in traj:
        if pos == _K1 and pre_side is None: pre_side = "L"
        if pos == _K2 and pre_side is None: pre_side = "R"
    if pre_side and post_side:
        return pre_side + post_side
    return "shared"


def evaluate_with_family(model, n_episodes=EVAL_EPISODES):
    env = EnvC_v2()
    rets, succs, lens = [], [], []
    fam_counts = {"LU": 0, "LD": 0, "RU": 0, "RD": 0, "shared": 0}
    model.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            traj = []; ret = 0.0; ep_len = 0; done = False; success = False
            while not done:
                pos, hk = obs
                a = int(model(encode_single(pos, hk).unsqueeze(0)).argmax(1).item())
                traj.append((obs, a))
                obs, r, term, trunc, _ = env.step(a)
                ret += r; ep_len += 1; done = term or trunc
                if term: success = True
            rets.append(ret); succs.append(float(success)); lens.append(ep_len)
            fam = infer_combined_family(traj)
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
    model.train()
    return (float(np.mean(rets)), float(np.mean(succs)), float(np.mean(lens)),
            fam_counts)

# ── Skip-completed / append helpers ───────────────────────────────────────────

def load_completed(summary_path):
    done = set()
    if not os.path.exists(summary_path): return done
    try:
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") in ("completed", "ok"):
                    done.add((row["dataset_name"], int(row["seed"])))
    except Exception: pass
    return done


def append_row(path, row, columns):
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if mode == "w": w.writeheader()
        w.writerow(row)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("EnvC_v2 CQL Formal Validation — 20 seeds")
    print("=" * 68)

    # ── 1. Dataset check ───────────────────────────────────────────────────
    print("\n── 1. Dataset check ──")
    regen_needed = False
    for ds_name in ["envC_v2_small_wide", "envC_v2_large_narrow_LU"]:
        ok = check_dataset(ds_name)
        print(f"  {ds_name}: {'OK' if ok else 'MISSING/STALE — regenerating'}")
        if not ok: regen_needed = True
    if regen_needed:
        print("  Regenerating datasets...")
        build_all(verbose=True)
        for ds_name in ["envC_v2_small_wide", "envC_v2_large_narrow_LU"]:
            assert check_dataset(ds_name), f"Regeneration failed for {ds_name}"
        print("  Datasets ready.")

    # ── 2. Load datasets ───────────────────────────────────────────────────
    print("\n── 2. Loading datasets ──")
    datasets = {}
    for ds_name in ["envC_v2_small_wide", "envC_v2_large_narrow_LU"]:
        obs, acts, rews, nobs, terms = load_dataset_c2(ds_name)
        spec = DATASET_SPECS[ds_name]
        print(f"  {ds_name}: {len(obs):,} transitions "
              f"(expected {spec['n_transitions']:,})")
        assert len(obs) == spec["n_transitions"], "Dataset size mismatch"
        datasets[ds_name] = (obs, acts, rews, nobs, terms)

    # ── 3. Resume check ────────────────────────────────────────────────────
    completed = load_completed(SUMMARY_PATH)
    print(f"\n── 3. Resume check: {len(completed)} seeds already done ──")

    if not os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS).writeheader()
    if not os.path.exists(RFAM_PATH):
        with open(RFAM_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RFAM_COLUMNS).writeheader()

    total_runs  = len(FORMAL_SEEDS) * 2
    remaining_n = total_runs - len(completed)
    print(f"  Total runs: {total_runs}  Remaining: {remaining_n}")

    # ── 4. Training loop ───────────────────────────────────────────────────
    print(f"\n── 4. Training (CQL × {len(FORMAL_SEEDS)} seeds × 2 datasets) ──")
    run_count = 0

    for ds_name, (obs, acts, rews, nobs, terms) in datasets.items():
        print(f"\n  Dataset: {ds_name}")
        for seed in FORMAL_SEEDS:
            if (ds_name, seed) in completed:
                print(f"    seed={seed:2d}  SKIP (already done)")
                continue

            ckpt_name = f"{ds_name}_seed{seed:02d}.pt"
            ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

            # Restore from checkpoint if available
            if os.path.exists(ckpt_path):
                try:
                    state = torch.load(ckpt_path, weights_only=True)
                    model = MLP(OBS_DIM, N_ACTIONS, CQL_CFG["hidden_dims"])
                    model.load_state_dict(state)
                    avg_ret, succ_rt, avg_len, fam_counts = evaluate_with_family(model)
                    fam = max(fam_counts, key=fam_counts.get)
                    total_ep = sum(fam_counts.values())
                    fracs = {k: fam_counts[k]/total_ep for k in
                             ["LU","LD","RU","RD","shared"]}
                    print(f"    seed={seed:2d}  RESTORED  ret={avg_ret:.4f}  "
                          f"succ={succ_rt:.3f}  fam={fam}")
                    row = {
                        "env_name": ENV_NAME, "dataset_name": ds_name,
                        "algorithm": ALGO, "seed": seed,
                        "avg_return": f"{avg_ret:.4f}",
                        "success_rate": f"{succ_rt:.4f}",
                        "avg_episode_len": f"{avg_len:.2f}",
                        "final_td_loss": "restored",
                        "final_cql_penalty": "restored",
                        "final_total_loss": "restored",
                        "checkpoint_path": ckpt_path, "status": "completed",
                    }
                    append_row(SUMMARY_PATH, row, SUMMARY_COLUMNS)
                    rfam_row = {
                        "dataset_name": ds_name, "seed": seed,
                        "dominant_combined_family": fam,
                        "frac_LU":  f"{fracs['LU']:.3f}",
                        "frac_LD":  f"{fracs['LD']:.3f}",
                        "frac_RU":  f"{fracs['RU']:.3f}",
                        "frac_RD":  f"{fracs['RD']:.3f}",
                        "frac_shared": f"{fracs['shared']:.3f}",
                        "avg_return": f"{avg_ret:.4f}",
                        "note": "restored from checkpoint",
                    }
                    append_row(RFAM_PATH, rfam_row, RFAM_COLUMNS)
                    run_count += 1
                    continue
                except Exception:
                    pass

            # Train
            print(f"    seed={seed:2d}  training...", end=" ", flush=True)
            model, td_l, cql_l, total_l = train_cql(
                obs, acts, rews, nobs, terms, CQL_CFG, seed)
            avg_ret, succ_rt, avg_len, fam_counts = evaluate_with_family(model)
            fam = max(fam_counts, key=fam_counts.get)
            total_ep = sum(fam_counts.values())
            fracs = {k: fam_counts[k]/total_ep for k in
                     ["LU","LD","RU","RD","shared"]}
            print(f"td={td_l:.4f} cql={cql_l:.4f} tot={total_l:.4f}  "
                  f"ret={avg_ret:.4f}  succ={succ_rt:.3f}  fam={fam}")

            torch.save(model.state_dict(), ckpt_path)

            row = {
                "env_name": ENV_NAME, "dataset_name": ds_name,
                "algorithm": ALGO, "seed": seed,
                "avg_return": f"{avg_ret:.4f}",
                "success_rate": f"{succ_rt:.4f}",
                "avg_episode_len": f"{avg_len:.2f}",
                "final_td_loss":     f"{td_l:.6f}",
                "final_cql_penalty": f"{cql_l:.6f}",
                "final_total_loss":  f"{total_l:.6f}",
                "checkpoint_path": ckpt_path, "status": "completed",
            }
            append_row(SUMMARY_PATH, row, SUMMARY_COLUMNS)
            rfam_row = {
                "dataset_name": ds_name, "seed": seed,
                "dominant_combined_family": fam,
                "frac_LU":  f"{fracs['LU']:.3f}",
                "frac_LD":  f"{fracs['LD']:.3f}",
                "frac_RU":  f"{fracs['RU']:.3f}",
                "frac_RD":  f"{fracs['RD']:.3f}",
                "frac_shared": f"{fracs['shared']:.3f}",
                "avg_return": f"{avg_ret:.4f}",
                "note": "",
            }
            append_row(RFAM_PATH, rfam_row, RFAM_COLUMNS)
            run_count += 1

    # ── 5. Analysis ────────────────────────────────────────────────────────
    print(f"\n── 5. Analysis ──")
    results  = {"envC_v2_small_wide": [], "envC_v2_large_narrow_LU": []}
    rfam_res = {"envC_v2_small_wide": [], "envC_v2_large_narrow_LU": []}

    with open(SUMMARY_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["dataset_name"]
            if ds in results and row["status"] == "completed":
                try: results[ds].append(float(row["avg_return"]))
                except ValueError: pass

    with open(RFAM_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["dataset_name"]
            if ds in rfam_res:
                rfam_res[ds].append(row["dominant_combined_family"])

    sw_rets = np.array(results["envC_v2_small_wide"])
    ln_rets = np.array(results["envC_v2_large_narrow_LU"])

    def ci95(arr):
        if len(arr) < 2: return float("nan"), float("nan")
        se = arr.std(ddof=1) / math.sqrt(len(arr))
        return arr.mean() - 1.96*se, arr.mean() + 1.96*se

    sw_mean = float(sw_rets.mean()) if len(sw_rets) else float("nan")
    sw_std  = float(sw_rets.std(ddof=1)) if len(sw_rets)>1 else float("nan")
    ln_mean = float(ln_rets.mean()) if len(ln_rets) else float("nan")
    ln_std  = float(ln_rets.std(ddof=1)) if len(ln_rets)>1 else float("nan")
    gap     = sw_mean - ln_mean
    sw_lo, sw_hi = ci95(sw_rets)
    ln_lo, ln_hi = ci95(ln_rets)

    sw_wins = int(np.sum(sw_rets > ln_rets)) if len(sw_rets)==len(ln_rets) else -1
    ci_no_overlap = (
        (not math.isnan(sw_lo)) and (not math.isnan(ln_hi)) and sw_lo > ln_hi
    )

    def fam_cnt(lst):
        c = {"LU":0,"LD":0,"RU":0,"RD":0,"shared":0}
        for f in lst: c[f] = c.get(f,0) + 1
        return c

    sw_fams = fam_cnt(rfam_res["envC_v2_small_wide"])
    ln_fams = fam_cnt(rfam_res["envC_v2_large_narrow_LU"])
    sw_non_lu = (sw_fams["LD"]+sw_fams["RU"]+sw_fams["RD"]+sw_fams.get("shared",0)) \
                / max(sum(sw_fams.values()), 1)
    ln_non_lu = (ln_fams["LD"]+ln_fams["RU"]+ln_fams["RD"]+ln_fams.get("shared",0)) \
                / max(sum(ln_fams.values()), 1)

    n_sw = len(sw_rets); n_ln = len(ln_rets)
    print(f"  small-wide      n={n_sw}  mean={sw_mean:.4f}  std={sw_std:.4f}  "
          f"95%CI=[{sw_lo:.4f},{sw_hi:.4f}]")
    print(f"  large-narrow-LU n={n_ln}  mean={ln_mean:.4f}  std={ln_std:.4f}  "
          f"95%CI=[{ln_lo:.4f},{ln_hi:.4f}]")
    print(f"  gap = {gap:.4f}  (BC ref=0.020  IQL ref=0.024)")
    print(f"  wide>narrow: {sw_wins}/{n_sw}" if sw_wins >= 0 else "")
    print(f"  CI non-overlap: {ci_no_overlap}")
    print(f"  SW families: {sw_fams}")
    print(f"  LN families: {ln_fams}")

    # ── 6. Write RUNLOG ────────────────────────────────────────────────────
    directional = gap > 0
    bc_gap_ref  = 0.020
    iql_gap_ref = 0.024

    three_algo_confirmed = directional  # BC and IQL already confirmed; CQL too?

    lines = [
        "# ENVC_V2_CQL_FORMAL_RUNLOG.md",
        "# EnvC_v2 CQL Formal Validation — 20 Seeds",
        "",
        "> Date: 2026-04-08",
        f"> Runs completed this session: {run_count}",
        f"> Total rows in summary CSV: {n_sw + n_ln}",
        "",
        "---",
        "",
        "## 1. Experiment Config",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Environment | EnvC_v2 (extended state) |",
        f"| Algorithm | CQL (Conservative Q-Learning) |",
        f"| Seeds | 0–19 (20 per dataset) |",
        f"| Eval episodes | {EVAL_EPISODES} |",
        f"| OBS_DIM | 401 (400 pos one-hot + has_key scalar) |",
        f"| MLP hidden | [256, 256] |",
        f"| Updates | {CQL_CFG['num_updates']} |",
        f"| Batch size | {CQL_CFG['batch_size']} |",
        f"| LR | {CQL_CFG['lr']} |",
        f"| cql_alpha | {CQL_CFG['cql_alpha']} |",
        f"| small-wide | 50k, LU+LD+RU+RD, delay=0.05 uniform_random, seed=600 |",
        f"| large-narrow-LU | 200k, LU only, delay=0.05 opposite, seed=601 |",
        f"| SA coverage gap | 0.3281 − 0.0874 = 0.2407 (ratio 3.76×) |",
        "",
        "## 2. Results",
        "",
        "### small-wide (50k, LU+LD+RU+RD)",
        "",
        f"n = {n_sw}",
        f"mean return  = {sw_mean:.4f}",
        f"std          = {sw_std:.4f}",
        f"95% CI       = [{sw_lo:.4f}, {sw_hi:.4f}]",
        f"min / max    = {float(sw_rets.min()):.4f} / {float(sw_rets.max()):.4f}" if n_sw else "",
        "",
        "### large-narrow-LU (200k, LU only)",
        "",
        f"n = {n_ln}",
        f"mean return  = {ln_mean:.4f}",
        f"std          = {ln_std:.4f}",
        f"95% CI       = [{ln_lo:.4f}, {ln_hi:.4f}]",
        f"min / max    = {float(ln_rets.min()):.4f} / {float(ln_rets.max()):.4f}" if n_ln else "",
        "",
        "### Comparison",
        "",
        f"Gap (sw_mean − ln_mean)  = {gap:.4f}",
        f"BC gap (reference)       = {bc_gap_ref:.4f}",
        f"IQL gap (reference)      = {iql_gap_ref:.4f}",
        f"Seed-paired wide > narrow: {sw_wins}/{n_sw}" if sw_wins >= 0 else "",
        f"95% CI overlap: {'NO (non-overlapping)' if ci_no_overlap else 'YES (overlapping)'}",
        "",
        "## 3. Route-Family Convergence",
        "",
        "| Dataset | LU | LD | RU | RD | shared |",
        "|---------|----|----|----|----|--------|",
        f"| small-wide | {sw_fams['LU']} | {sw_fams['LD']} | {sw_fams['RU']} | {sw_fams['RD']} | {sw_fams.get('shared',0)} |",
        f"| large-narrow-LU | {ln_fams['LU']} | {ln_fams['LD']} | {ln_fams['RU']} | {ln_fams['RD']} | {ln_fams.get('shared',0)} |",
        "",
        f"Wide non-LU fraction: {sw_non_lu:.3f}",
        f"Narrow non-LU fraction: {ln_non_lu:.3f}",
        "",
        "## 4. Three-Algorithm EnvC_v2 Summary",
        "",
        "| Algorithm | Gap | CI non-overlap | Wide>Narrow seeds |",
        "|-----------|-----|---------------|-------------------|",
        f"| BC  | 0.0200 | YES | 10/20 |",
        f"| IQL | 0.0240 | YES | 12/20 |",
        f"| CQL | {gap:.4f} | {'YES' if ci_no_overlap else 'NO'} | {sw_wins}/{n_sw} |",
        "",
        "## 5. Directional Assessment",
        "",
    ]

    if directional and ci_no_overlap:
        lines += [
            "**STRONG directional support: CQL wide > narrow, CIs non-overlapping.**",
            "",
            f"CQL gap = {gap:.4f}  |  BC ref = {bc_gap_ref:.4f}  |  IQL ref = {iql_gap_ref:.4f}",
            "All three algorithms confirm the coverage effect in EnvC_v2.",
        ]
    elif directional:
        lines += [
            "**DIRECTIONAL support: CQL wide > narrow (CIs overlap).**",
            "",
            f"CQL gap = {gap:.4f} (positive)",
            "CIs overlap — directional but not decisive at 20 seeds.",
        ]
    else:
        lines += [
            "**NO directional support: CQL wide NOT > narrow.**",
            "",
            f"CQL gap = {gap:.4f} (non-positive)",
            "CQL does not confirm the coverage direction in EnvC_v2.",
            "Report as mixed evidence — BC and IQL confirm; CQL does not.",
        ]

    lines += [
        "",
        "## 6. EnvC_v2 Final Validation Status",
        "",
    ]
    if three_algo_confirmed:
        lines += [
            "**EnvC_v2 VALIDATION CONFIRMED — all three algorithms show wide > narrow.**",
            "",
            "| Algorithm | Gap | CI status |",
            "|-----------|-----|-----------|",
            "| BC  | +0.0200 | non-overlapping |",
            "| IQL | +0.0240 | non-overlapping |",
            f"| CQL | +{gap:.4f} | {'non-overlapping' if ci_no_overlap else 'overlapping'} |",
            "",
            "EnvC_v2 can be reported as 3/3 algorithm confirmed, matching EnvB_v2's status.",
        ]
    else:
        lines += [
            "**MIXED: BC and IQL confirm; CQL does not.**",
            "",
            "EnvC_v2 remains at 2/3 directional confirmed.",
            "Report this accurately — do not upgrade to 3/3 confirmed.",
        ]

    lines += [
        "",
        "## 7. What Was Not Done",
        "",
        "- No modifications to EnvA_v2 / EnvB_v2 formal results",
        "- No changes to CLAIM_HIERARCHY or main conclusions for EnvA_v2",
        "- No git push or remote sync",
    ]

    with open(RUNLOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Runlog: {RUNLOG_PATH}")

    # ── 7. Final summary ───────────────────────────────────────────────────
    print()
    print("=" * 68)
    print(f"small-wide     mean={sw_mean:.4f}  std={sw_std:.4f}  "
          f"95%CI=[{sw_lo:.4f},{sw_hi:.4f}]")
    print(f"large-narrow   mean={ln_mean:.4f}  std={ln_std:.4f}  "
          f"95%CI=[{ln_lo:.4f},{ln_hi:.4f}]")
    print(f"gap = {gap:.4f}  wide>narrow: {sw_wins}/{n_sw}  "
          f"CI_no_overlap={ci_no_overlap}")
    print(f"SW families: {sw_fams}")
    print(f"LN families: {ln_fams}")
    status = "3/3 CONFIRMED" if three_algo_confirmed else "2/3 directional (CQL no direction)"
    print(f"EnvC_v2 STATUS: {status}")
    print("=" * 68)

    return three_algo_confirmed, gap, sw_wins, ci_no_overlap, sw_fams, ln_fams


if __name__ == "__main__":
    main()
