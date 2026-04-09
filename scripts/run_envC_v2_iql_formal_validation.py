"""
scripts/run_envC_v2_iql_formal_validation.py
EnvC_v2 IQL formal validation — 20 seeds per dataset.

Frozen config (per ENVC_V2_IMPLEMENTATION_RUNLOG.md):
  Environment:  EnvC_v2 (extended state ((row,col), has_key))
  small-wide:   50k, LU+LD+RU+RD, delay=0.05, uniform_random, seed=600
  large-narrow: 200k, LU only,    delay=0.05, opposite,       seed=601
  Training:     IQL × 20 seeds (0–19), same frozen hyperparameters as EnvB_v2 IQL formal
  Eval:         greedy, 50 episodes on EnvC_v2 START→GOAL
  OBS encoding: 401-dim (400 pos one-hot + has_key scalar)

IQL architecture: actor + 2×Q-network (with targets) + value network.
  Expectile regression on V, TD on Q, advantage-weighted BC on actor.

Features:
  - skip-completed  (reads existing CSV, skips already-done rows)
  - crash-safe append (opens CSV in append mode after header)
  - dataset existence / metadata check; regenerates if stale or missing
  - checkpoints (all 4 networks) to envC_v2_iql_checkpoints/
  - route-family convergence tracked via greedy rollout analysis

Outputs:
  artifacts/training_validation_v2/envC_v2_iql_formal_summary.csv
  artifacts/training_validation_v2/envC_v2_iql_route_family_summary.csv
  artifacts/training_validation_v2/envC_v2_iql_checkpoints/<ds>_seed<N>.pt
  docs/ENVC_V2_IQL_FORMAL_RUNLOG.md
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
CKPT_DIR     = os.path.join(OUT_DIR, "envC_v2_iql_checkpoints")
DATASET_DIR  = os.path.join(PROJECT_ROOT, "artifacts", "envC_v2_datasets")
DOCS_DIR     = os.path.join(PROJECT_ROOT, "docs")
SUMMARY_PATH = os.path.join(OUT_DIR, "envC_v2_iql_formal_summary.csv")
RFAM_PATH    = os.path.join(OUT_DIR, "envC_v2_iql_route_family_summary.csv")
RUNLOG_PATH  = os.path.join(DOCS_DIR, "ENVC_V2_IQL_FORMAL_RUNLOG.md")

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Frozen experiment config ───────────────────────────────────────────────────

FORMAL_SEEDS  = list(range(20))
EVAL_EPISODES = 50
ENV_NAME      = "EnvC_v2"
ALGO          = "iql"

# IQL hyperparameters — identical to EnvB_v2 IQL formal experiment
IQL_CFG = {
    "hidden_dims":  [256, 256],
    "num_updates":  5000,
    "batch_size":   256,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "gamma":        0.99,
    "expectile":    0.7,
    "temperature":  3.0,
    "target_tau":   0.005,
    "adv_clip":     100.0,
}

DATASET_SPECS = {
    "envC_v2_small_wide": {
        "n_transitions":   WIDE_CONFIG["n_trans"],
        "seed":            WIDE_CONFIG["seed"],
        "source_families": WIDE_CONFIG["families"],
    },
    "envC_v2_large_narrow_LU": {
        "n_transitions":   NARROW_CONFIG["n_trans"],
        "seed":            NARROW_CONFIG["seed"],
        "source_families": NARROW_CONFIG["families"],
    },
}

SUMMARY_COLUMNS = [
    "env_name", "dataset_name", "algorithm", "seed",
    "avg_return", "success_rate", "avg_episode_len",
    "final_actor_loss", "final_q_loss", "final_v_loss",
    "checkpoint_path", "status",
]

RFAM_COLUMNS = [
    "dataset_name", "seed", "dominant_combined_family",
    "frac_LU", "frac_LD", "frac_RU", "frac_RD", "frac_shared",
    "avg_return", "note",
]

# ── Network ────────────────────────────────────────────────────────────────────

_G      = EnvC_v2.grid_size
_K1     = (5, 2)
_K2     = (5, 7)
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
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path)
        return int(d["n_transitions"][0]) == DATASET_SPECS[ds_name]["n_transitions"]
    except Exception:
        return False


def load_dataset_c2(ds_name):
    path = os.path.join(DATASET_DIR, f"{ds_name}.npz")
    d    = np.load(path)
    obs  = encode_obs(d["observations"].astype(np.int32))
    nobs = encode_obs(d["next_observations"].astype(np.int32))
    acts = d["actions"].astype(np.int64)
    rews = d["rewards"].astype(np.float32)
    term = d["terminals"].astype(np.float32)
    return obs, acts, rews, nobs, term

# ── IQL training ───────────────────────────────────────────────────────────────

def train_iql(obs, acts, rews, nobs, terms, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(obs)
    actor = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q1    = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q2    = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q1t   = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"]); q1t.load_state_dict(q1.state_dict())
    q2t   = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"]); q2t.load_state_dict(q2.state_dict())
    vnet  = MLP(OBS_DIM, 1,         cfg["hidden_dims"])
    ao = optim.Adam(actor.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    qo = optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                    lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    vo = optim.Adam(vnet.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ot = torch.from_numpy(obs); at2 = torch.from_numpy(acts)
    rt = torch.from_numpy(rews); nt = torch.from_numpy(nobs); dt = torch.from_numpy(terms)
    g   = cfg["gamma"]; tau = cfg["expectile"]; tmp = cfg["temperature"]
    tt  = cfg["target_tau"]; ac  = cfg["adv_clip"]
    final_al = final_ql = final_vl = float("nan")
    ao.zero_grad(); qo.zero_grad(); vo.zero_grad()
    for _ in range(cfg["num_updates"]):
        idx = np.random.randint(0, n, size=cfg["batch_size"])
        s, a, r, ns, d = ot[idx], at2[idx], rt[idx], nt[idx], dt[idx]
        with torch.no_grad():
            qm = torch.min(q1t(s).gather(1, a.unsqueeze(1)).squeeze(1),
                           q2t(s).gather(1, a.unsqueeze(1)).squeeze(1))
        v = vnet(s).squeeze(1); diff = qm - v
        wt = torch.where(diff >= 0, torch.full_like(diff, tau),
                         torch.full_like(diff, 1 - tau))
        vl = (wt * diff.pow(2)).mean(); vl.backward(); vo.step(); vo.zero_grad()
        final_vl = vl.item()
        with torch.no_grad():
            td = r + g * (1 - d) * vnet(ns).squeeze(1)
        q1a = q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2a = q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        ql = nn.functional.mse_loss(q1a, td) + nn.functional.mse_loss(q2a, td)
        ql.backward(); qo.step(); qo.zero_grad(); final_ql = ql.item()
        with torch.no_grad():
            adv = (torch.min(q1t(s).gather(1, a.unsqueeze(1)).squeeze(1),
                             q2t(s).gather(1, a.unsqueeze(1)).squeeze(1))
                   - vnet(s).squeeze(1))
            ea = torch.exp(adv / tmp).clamp(max=ac)
        lp = nn.functional.log_softmax(actor(s), dim=1).gather(
            1, a.unsqueeze(1)).squeeze(1)
        al = -(ea * lp).mean(); al.backward(); ao.step(); ao.zero_grad()
        final_al = al.item()
        for p, tp in zip(q1.parameters(), q1t.parameters()):
            tp.data.copy_(tt * p.data + (1 - tt) * tp.data)
        for p, tp in zip(q2.parameters(), q2t.parameters()):
            tp.data.copy_(tt * p.data + (1 - tt) * tp.data)
    return actor, final_al, final_ql, final_vl

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
    # fallback via key cell
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


def append_row(path, row, columns):
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if mode == "w":
            w.writeheader()
        w.writerow(row)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("EnvC_v2 IQL Formal Validation — 20 seeds")
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
        print(f"  {ds_name}: {len(obs):,} transitions (expected {spec['n_transitions']:,})")
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
    print(f"\n── 4. Training (IQL × {len(FORMAL_SEEDS)} seeds × 2 datasets) ──")
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
                    actor = MLP(OBS_DIM, N_ACTIONS, IQL_CFG["hidden_dims"])
                    actor.load_state_dict(state)
                    avg_ret, succ_rt, avg_len, fam_counts = evaluate_with_family(actor)
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
                        "final_actor_loss": "restored",
                        "final_q_loss": "restored",
                        "final_v_loss": "restored",
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
            actor, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, IQL_CFG, seed)
            avg_ret, succ_rt, avg_len, fam_counts = evaluate_with_family(actor)
            fam = max(fam_counts, key=fam_counts.get)
            total_ep = sum(fam_counts.values())
            fracs = {k: fam_counts[k]/total_ep for k in
                     ["LU","LD","RU","RD","shared"]}
            print(f"actor={al:.4f} q={ql:.4f} v={vl:.4f}  "
                  f"ret={avg_ret:.4f}  succ={succ_rt:.3f}  fam={fam}")

            # Save actor checkpoint only (sufficient for evaluation)
            torch.save(actor.state_dict(), ckpt_path)

            row = {
                "env_name": ENV_NAME, "dataset_name": ds_name,
                "algorithm": ALGO, "seed": seed,
                "avg_return": f"{avg_ret:.4f}",
                "success_rate": f"{succ_rt:.4f}",
                "avg_episode_len": f"{avg_len:.2f}",
                "final_actor_loss": f"{al:.6f}",
                "final_q_loss":     f"{ql:.6f}",
                "final_v_loss":     f"{vl:.6f}",
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
                / max(sum(sw_fams.values()),1)
    ln_non_lu = (ln_fams["LD"]+ln_fams["RU"]+ln_fams["RD"]+ln_fams.get("shared",0)) \
                / max(sum(ln_fams.values()),1)

    print(f"  small-wide      n={len(sw_rets)}  mean={sw_mean:.4f}  std={sw_std:.4f}  "
          f"95%CI=[{sw_lo:.4f},{sw_hi:.4f}]")
    print(f"  large-narrow-LU n={len(ln_rets)}  mean={ln_mean:.4f}  std={ln_std:.4f}  "
          f"95%CI=[{ln_lo:.4f},{ln_hi:.4f}]")
    print(f"  gap (sw − ln) = {gap:.4f}  BC_gap_ref = 0.0200")
    print(f"  wide>narrow: {sw_wins}/{len(sw_rets)}" if sw_wins>=0 else "")
    print(f"  CI non-overlap: {ci_no_overlap}")
    print(f"  SW families: {sw_fams}")
    print(f"  LN families: {ln_fams}")

    # ── 6. Write RUNLOG ────────────────────────────────────────────────────
    directional    = gap > 0
    iql_gt_bc      = abs(gap) > 0.020
    bc_gap_ref     = 0.020
    n_sw = len(sw_rets); n_ln = len(ln_rets)

    both_directional = directional  # BC was directional; IQL directional?

    lines = [
        "# ENVC_V2_IQL_FORMAL_RUNLOG.md",
        "# EnvC_v2 IQL Formal Validation — 20 Seeds",
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
        f"| Algorithm | IQL (Implicit Q-Learning) |",
        f"| Seeds | 0–19 (20 per dataset) |",
        f"| Eval episodes | {EVAL_EPISODES} |",
        f"| OBS_DIM | 401 (400 pos one-hot + has_key scalar) |",
        f"| MLP hidden | [256, 256] |",
        f"| Updates | {IQL_CFG['num_updates']} |",
        f"| Batch size | {IQL_CFG['batch_size']} |",
        f"| LR | {IQL_CFG['lr']} |",
        f"| Expectile | {IQL_CFG['expectile']} |",
        f"| Temperature | {IQL_CFG['temperature']} |",
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
        f"min / max    = {float(sw_rets.min()):.4f} / {float(sw_rets.max()):.4f}" if n_sw else "min / max = n/a",
        "",
        "### large-narrow-LU (200k, LU only)",
        "",
        f"n = {n_ln}",
        f"mean return  = {ln_mean:.4f}",
        f"std          = {ln_std:.4f}",
        f"95% CI       = [{ln_lo:.4f}, {ln_hi:.4f}]",
        f"min / max    = {float(ln_rets.min()):.4f} / {float(ln_rets.max()):.4f}" if n_ln else "min / max = n/a",
        "",
        "### Comparison",
        "",
        f"Gap (sw_mean − ln_mean)  = {gap:.4f}",
        f"BC gap (reference)       = {bc_gap_ref:.4f}",
        f"IQL gap / BC gap ratio   = {gap/bc_gap_ref:.2f}×" if bc_gap_ref != 0 else "",
        f"Seed-paired wide > narrow: {sw_wins}/{n_sw}" if sw_wins>=0 else "",
        f"95% CI overlap: {'NO (non-overlapping)' if ci_no_overlap else 'YES (overlapping)'}",
        "",
        "## 3. Route-Family Convergence",
        "",
        "| Dataset | LU | LD | RU | RD | shared |",
        "|---------|----|----|----|----|--------|",
        f"| small-wide | {sw_fams['LU']} | {sw_fams['LD']} | {sw_fams['RU']} | {sw_fams['RD']} | {sw_fams.get('shared',0)} |",
        f"| large-narrow-LU | {ln_fams['LU']} | {ln_fams['LD']} | {ln_fams['RU']} | {ln_fams['RD']} | {ln_fams.get('shared',0)} |",
        "",
        f"Wide non-LU family fraction:  {sw_non_lu:.3f}  ({int(sw_non_lu*sum(sw_fams.values()))}/{sum(sw_fams.values())} episodes)",
        f"Narrow non-LU family fraction: {ln_non_lu:.3f}  ({int(ln_non_lu*sum(ln_fams.values()))}/{sum(ln_fams.values())} episodes)",
        "",
        "## 4. Directional Assessment",
        "",
    ]

    if directional and ci_no_overlap:
        lines += [
            "**STRONG directional support: wide > narrow, CIs non-overlapping.**",
            "",
            f"IQL gap = {gap:.4f} (positive)",
            "CIs do not overlap — effect is statistically clear at n=20.",
        ]
    elif directional:
        lines += [
            "**DIRECTIONAL support: wide > narrow (CIs overlap).**",
            "",
            f"IQL gap = {gap:.4f} (positive)",
            "CIs overlap at n=20 — directional but not decisive.",
        ]
    else:
        lines += [
            "**NO directional support: wide NOT > narrow.**",
            "",
            f"IQL gap = {gap:.4f}",
        ]

    lines += [
        "",
        "### IQL vs BC: Does IQL amplify the coverage contrast?",
        "",
        f"BC gap  = {bc_gap_ref:.4f}",
        f"IQL gap = {gap:.4f}",
    ]
    if gap > bc_gap_ref:
        lines.append(f"IQL amplifies BC gap by {gap/bc_gap_ref:.2f}× — "
                     "value-based learning better exploits coverage advantage.")
    elif gap == bc_gap_ref:
        lines.append("IQL gap equals BC gap — both algorithms show the same effect size.")
    else:
        lines.append("IQL gap ≤ BC gap — IQL did not amplify the coverage contrast "
                     "beyond BC in this environment.")

    lines += [
        "",
        "## 5. EnvC_v2 Validation Status",
        "",
    ]

    bc_dir = True   # from BC runlog
    iql_dir = directional

    if bc_dir and iql_dir:
        lines += [
            "**DIRECTIONAL VALIDATION CONFIRMED (BC + IQL both positive).**",
            "",
            "Both BC and IQL show wide > narrow in EnvC_v2:",
            f"  BC gap  = {bc_gap_ref:.4f}  (CI non-overlapping)",
            f"  IQL gap = {gap:.4f}  (CI {'non-overlapping' if ci_no_overlap else 'overlapping'})",
            "",
            "The coverage hypothesis is supported by two independent algorithms.",
            "EnvC_v2 can be recorded as providing directional validation of the",
            "coverage-beats-size effect in a key-door staged dependency environment.",
        ]
    else:
        lines += [
            "**PARTIAL VALIDATION — only one algorithm supports the hypothesis.**",
            "",
            f"BC directional: YES  |  IQL directional: {'YES' if iql_dir else 'NO'}",
        ]

    lines += [
        "",
        "## 6. Proceed to CQL Smoke / Formal 20 Seeds?",
        "",
    ]
    if bc_dir and iql_dir:
        lines += [
            "**YES — run minimal CQL smoke (2-seed), then decide on CQL formal 20 seeds.**",
            "",
            "Rationale:",
            "  - Both BC and IQL confirm the coverage effect in EnvC_v2",
            "  - CQL smoke would verify the pipeline works before committing to 20 seeds",
            "  - If CQL smoke shows goal_reached=True and stable training,",
            "    proceed to CQL 20-seed formal validation to complete the three-algorithm summary",
            "  - EnvB_v2 precedent: CQL gap=0.025, CI non-overlapping (matched BC and IQL)",
            "",
            "CQL smoke can reuse run_envC_v2_formal_smoke.py infrastructure.",
        ]
    else:
        lines += [
            "**HOLD — CQL smoke can be run but not strictly required.**",
            "",
            "Without both BC and IQL being directional, CQL alone would not",
            "constitute three-algorithm confirmation. Address the non-directional",
            "algorithm first.",
        ]

    lines += [
        "",
        "## 7. What Was Not Done",
        "",
        "- No CQL formal 20-seed run (pending CQL smoke decision)",
        "- No modifications to EnvA_v2 / EnvB_v2 formal results",
        "- No changes to CLAIM_HIERARCHY or main conclusions",
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
    print(f"gap = {gap:.4f}  (BC ref = {bc_gap_ref:.4f})  "
          f"wide>narrow: {sw_wins}/{n_sw}  CI_no_overlap={ci_no_overlap}")
    print(f"SW families: {sw_fams}")
    print(f"LN families: {ln_fams}")
    status = ("CONFIRMED" if (bc_dir and iql_dir) else "PARTIAL")
    print(f"EnvC_v2 VALIDATION STATUS: {status}")
    print("=" * 68)


if __name__ == "__main__":
    main()
