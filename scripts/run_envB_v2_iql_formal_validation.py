"""
scripts/run_envB_v2_iql_formal_validation.py
EnvB_v2 IQL formal validation — 20 seeds per dataset.

Frozen config (per ENVB_V2_PRE_FORMAL_DECISION.md):
  Environment:    EnvB_v2
  small-wide:     50k,  families A+B+C, delay=0.25, seed=425
  large-narrow-A: 200k, family A only,  delay=0.10, seed=411
  Training:       IQL × 20 seeds (0-19), same hyperparameters as EnvA_v2 IQL
  Eval:           greedy, 50 episodes on EnvB_v2 START→GOAL

Frozen IQL config (from run_envA_v2_iql_sanity.py):
  hidden_dims=[256,256], num_updates=5000, batch_size=256,
  expectile=0.7, temperature=3.0, gamma=0.99,
  actor_lr=critic_lr=value_lr=3e-4, weight_decay=1e-4,
  target_tau=0.005, adv_clip=100.0

Features:
  - skip-completed (reads existing CSV, skips already-done seeds)
  - crash-safe append
  - dataset existence / metadata check; regenerates if stale
  - checkpoints: all 4 networks saved per seed
  - route-family convergence tracking during greedy eval

Outputs:
  artifacts/training_validation_v2/envB_v2_iql_formal_summary.csv
  artifacts/training_validation_v2/envB_v2_iql_route_family_summary.csv
  artifacts/training_validation_v2/envB_v2_iql_checkpoints/
  docs/ENVB_V2_IQL_FORMAL_RUNLOG.md
"""

import sys, os, csv, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from envs.gridworld_envs import EnvB_v2, N_ACTIONS, CORRIDOR_COLS_B2
from scripts.build_envB_v2_datasets import (
    build_all, WIDE_CONFIG, NARROW_CONFIG,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR       = os.path.join(PROJECT_ROOT, "artifacts", "training_validation_v2")
CKPT_DIR      = os.path.join(OUT_DIR, "envB_v2_iql_checkpoints")
DATASET_DIR   = os.path.join(PROJECT_ROOT, "artifacts", "envB_v2_datasets")
DOCS_DIR      = os.path.join(PROJECT_ROOT, "docs")
SUMMARY_PATH  = os.path.join(OUT_DIR, "envB_v2_iql_formal_summary.csv")
ROUTE_PATH    = os.path.join(OUT_DIR, "envB_v2_iql_route_family_summary.csv")
RUNLOG_PATH   = os.path.join(DOCS_DIR, "ENVB_V2_IQL_FORMAL_RUNLOG.md")

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Frozen experiment config ───────────────────────────────────────────────────

FORMAL_SEEDS  = list(range(20))
EVAL_EPISODES = 50
ENV_NAME      = "EnvB_v2"
ALGO          = "iql"

# IQL hyperparameters — identical to run_envA_v2_iql_sanity.IQL_CFG
IQL_CFG = {
    "hidden_dims":  [256, 256],
    "batch_size":   256,
    "num_updates":  5000,
    "gamma":        0.99,
    "expectile":    0.7,
    "temperature":  3.0,
    "actor_lr":     3e-4,
    "critic_lr":    3e-4,
    "value_lr":     3e-4,
    "weight_decay": 1e-4,
    "target_tau":   0.005,
    "adv_clip":     100.0,
}

DATASET_SPECS = {
    "envB_v2_small_wide": {
        "n_transitions": WIDE_CONFIG["n_trans"],
        "seed":          WIDE_CONFIG["seed"],
    },
    "envB_v2_large_narrow_A": {
        "n_transitions": NARROW_CONFIG["n_trans"],
        "seed":          NARROW_CONFIG["seed"],
    },
}

SUMMARY_COLUMNS = [
    "env_name", "dataset_name", "algorithm", "seed",
    "avg_return", "success_rate", "avg_episode_len",
    "final_actor_loss", "final_q_loss", "final_v_loss",
    "checkpoint_path", "status",
]

ROUTE_COLUMNS = [
    "dataset_name", "seed", "dominant_family",
    "frac_A", "frac_B", "frac_C", "frac_stem",
    "avg_return", "note",
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
    path = os.path.join(DATASET_DIR, f"{ds_name}.npz")
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path)
        return int(d["n_transitions"][0]) == DATASET_SPECS[ds_name]["n_transitions"]
    except Exception:
        return False


def load_dataset_b2(ds_name):
    path = os.path.join(DATASET_DIR, f"{ds_name}.npz")
    d = np.load(path)
    obs   = encode_obs(d["observations"].astype(np.int32))
    nobs  = encode_obs(d["next_observations"].astype(np.int32))
    acts  = d["actions"].astype(np.int64)
    rews  = d["rewards"].astype(np.float32)
    terms = d["terminals"].astype(np.float32)
    return obs, acts, rews, nobs, terms

# ── Route-family classification ───────────────────────────────────────────────

def classify_pos(r, c):
    """Return family label for a grid position."""
    if 4 <= r <= 15:
        for fam, (lo, hi) in CORRIDOR_COLS_B2.items():
            if lo <= c <= hi:
                return fam
    return "stem"


def infer_dominant_family(pos_trace):
    """Given a list of (r,c) positions, return the corridor family visited most."""
    counts = {"A": 0, "B": 0, "C": 0, "stem": 0}
    for r, c in pos_trace:
        counts[classify_pos(r, c)] += 1
    total = len(pos_trace)
    if total == 0:
        return "fail", 0.0, 0.0, 0.0, 0.0
    fracs = {k: v / total for k, v in counts.items()}
    corridor_counts = {k: v for k, v in counts.items() if k in ("A", "B", "C")}
    if sum(corridor_counts.values()) == 0:
        return "fail", fracs["A"], fracs["B"], fracs["C"], fracs["stem"]
    dominant = max(corridor_counts, key=corridor_counts.get)
    return dominant, fracs["A"], fracs["B"], fracs["C"], fracs["stem"]

# ── IQL training ───────────────────────────────────────────────────────────────

def train_iql(obs, acts, rews, nobs, terms, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(obs)

    actor     = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q1        = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q2        = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q1_tgt    = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q2_tgt    = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    value_net = MLP(OBS_DIM, 1,         cfg["hidden_dims"])

    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(),
                           lr=cfg["actor_lr"],  weight_decay=cfg["weight_decay"])
    q_opt     = optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                           lr=cfg["critic_lr"], weight_decay=cfg["weight_decay"])
    v_opt     = optim.Adam(value_net.parameters(),
                           lr=cfg["value_lr"],  weight_decay=cfg["weight_decay"])

    obs_t   = torch.from_numpy(obs);   acts_t  = torch.from_numpy(acts)
    rews_t  = torch.from_numpy(rews);  nobs_t  = torch.from_numpy(nobs)
    terms_t = torch.from_numpy(terms)

    gamma = cfg["gamma"]; tau = cfg["expectile"]
    temperature = cfg["temperature"]; adv_clip = cfg["adv_clip"]
    target_tau = cfg["target_tau"]

    final_al = final_ql = final_vl = float("nan")

    for _ in range(cfg["num_updates"]):
        idx = np.random.randint(0, n, size=cfg["batch_size"])
        s, a, r, ns, done = (obs_t[idx], acts_t[idx], rews_t[idx],
                              nobs_t[idx], terms_t[idx])

        # Value network — expectile regression on Q_min(s,a)
        with torch.no_grad():
            q_min_sa = torch.min(
                q1_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1),
                q2_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1))
        v    = value_net(s).squeeze(1)
        diff = q_min_sa - v
        wt   = torch.where(diff >= 0,
                           torch.full_like(diff, tau),
                           torch.full_like(diff, 1.0 - tau))
        vl = (wt * diff.pow(2)).mean()
        v_opt.zero_grad(); vl.backward(); v_opt.step()
        final_vl = vl.item()

        # Twin Q networks — Bellman backup through V(s')
        with torch.no_grad():
            td = r + gamma * (1.0 - done) * value_net(ns).squeeze(1)
        q1a = q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2a = q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        ql = nn.functional.mse_loss(q1a, td) + nn.functional.mse_loss(q2a, td)
        q_opt.zero_grad(); ql.backward(); q_opt.step()
        final_ql = ql.item()

        # Actor — advantage-weighted behavioral cloning
        with torch.no_grad():
            adv = (torch.min(q1_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1),
                             q2_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1))
                   - value_net(s).squeeze(1))
            exp_adv = torch.exp(adv / temperature).clamp(max=adv_clip)
        log_pi = nn.functional.log_softmax(actor(s), dim=1).gather(
            1, a.unsqueeze(1)).squeeze(1)
        al = -(exp_adv * log_pi).mean()
        actor_opt.zero_grad(); al.backward(); actor_opt.step()
        final_al = al.item()

        # Soft target update
        for p, tp in zip(q1.parameters(), q1_tgt.parameters()):
            tp.data.copy_(target_tau * p.data + (1 - target_tau) * tp.data)
        for p, tp in zip(q2.parameters(), q2_tgt.parameters()):
            tp.data.copy_(target_tau * p.data + (1 - target_tau) * tp.data)

    return actor, q1, q2, value_net, final_al, final_ql, final_vl

# ── Evaluation with route-family tracking ─────────────────────────────────────

def evaluate_with_routes(actor, n_episodes=EVAL_EPISODES):
    """Greedy rollout on EnvB_v2; returns perf metrics + route family data."""
    env = EnvB_v2()
    returns, succs, lens = [], [], []
    all_traces = []   # per-episode position traces
    actor.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs_raw, _ = env.reset()
            ret = 0.0; ep_len = 0; done = False; success = False
            trace = [obs_raw]
            while not done:
                logits = actor(encode_single(*obs_raw).unsqueeze(0))
                a = int(logits.argmax(1).item())
                obs_raw, r, term, trunc, _ = env.step(a)
                ret += r; ep_len += 1; done = term or trunc
                trace.append(obs_raw)
                if term: success = True
            returns.append(ret); succs.append(float(success)); lens.append(ep_len)
            all_traces.append(trace)
    actor.train()

    # Aggregate route-family usage across all episodes
    all_positions = [pos for trace in all_traces for pos in trace]
    dominant, fA, fB, fC, fS = infer_dominant_family(all_positions)

    return (float(np.mean(returns)), float(np.mean(succs)), float(np.mean(lens)),
            dominant, fA, fB, fC, fS)

# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_iql_ckpt(ckpt_path, actor, q1, q2, value_net):
    torch.save({
        "actor":     actor.state_dict(),
        "q1":        q1.state_dict(),
        "q2":        q2.state_dict(),
        "value_net": value_net.state_dict(),
    }, ckpt_path)


def load_iql_ckpt(ckpt_path):
    """Load IQL checkpoint; return actor model or raise on failure."""
    state = torch.load(ckpt_path, weights_only=True)
    actor = MLP(OBS_DIM, N_ACTIONS, IQL_CFG["hidden_dims"])
    actor.load_state_dict(state["actor"])
    return actor

# ── Skip-completed helpers ─────────────────────────────────────────────────────

def load_completed(summary_path):
    done = set()
    if not os.path.exists(summary_path):
        return done
    try:
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "completed":
                    done.add((row["dataset_name"], int(row["seed"])))
    except Exception:
        pass
    return done


def append_row(path, row, columns, write_header=False):
    mode = "w" if write_header or not os.path.exists(path) else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if mode == "w":
            w.writeheader()
        w.writerow(row)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("EnvB_v2 IQL Formal Validation — 20 seeds")
    print("=" * 68)

    # ── 1. Dataset check ──────────────────────────────────────────────────
    print("\n── 1. Dataset check ──")
    regen = False
    for ds in ["envB_v2_small_wide", "envB_v2_large_narrow_A"]:
        ok = check_dataset(ds)
        print(f"  {ds}: {'OK' if ok else 'MISSING/STALE'}")
        if not ok: regen = True
    if regen:
        print("  Regenerating datasets...")
        build_all(verbose=True)

    # ── 2. Load datasets ───────────────────────────────────────────────────
    print("\n── 2. Loading datasets ──")
    datasets = {}
    for ds_name in ["envB_v2_small_wide", "envB_v2_large_narrow_A"]:
        obs, acts, rews, nobs, terms = load_dataset_b2(ds_name)
        assert len(obs) == DATASET_SPECS[ds_name]["n_transitions"]
        print(f"  {ds_name}: {len(obs)} transitions  OK")
        datasets[ds_name] = (obs, acts, rews, nobs, terms)

    # ── 3. Resume check ────────────────────────────────────────────────────
    completed = load_completed(SUMMARY_PATH)
    total_runs = len(FORMAL_SEEDS) * 2
    print(f"\n── 3. Resume: {len(completed)} done, {total_runs - len(completed)} remaining ──")

    # Init summary CSV header if new
    if not os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS).writeheader()

    # Route-family results collected in memory; written at end
    route_rows = []

    # ── 4. Training loop ───────────────────────────────────────────────────
    print(f"\n── 4. Training (IQL × {len(FORMAL_SEEDS)} seeds × 2 datasets) ──")
    run_count = 0

    for ds_name, (obs, acts, rews, nobs, terms) in datasets.items():
        print(f"\n  Dataset: {ds_name}")
        for seed in FORMAL_SEEDS:
            if (ds_name, seed) in completed:
                print(f"    seed={seed:2d}  SKIP")
                # Still need route data for completed seeds
                ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_seed{seed:02d}.pt")
                if os.path.exists(ckpt_path):
                    try:
                        actor = load_iql_ckpt(ckpt_path)
                        _, _, _, dom, fA, fB, fC, fS = evaluate_with_routes(actor)
                        route_rows.append({
                            "dataset_name": ds_name, "seed": seed,
                            "dominant_family": dom,
                            "frac_A": f"{fA:.3f}", "frac_B": f"{fB:.3f}",
                            "frac_C": f"{fC:.3f}", "frac_stem": f"{fS:.3f}",
                            "avg_return": "see_summary", "note": "restored",
                        })
                    except Exception:
                        pass
                continue

            ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_seed{seed:02d}.pt")

            # Try restoring from existing checkpoint
            if os.path.exists(ckpt_path):
                try:
                    actor = load_iql_ckpt(ckpt_path)
                    avg_ret, succ_rt, avg_len, dom, fA, fB, fC, fS = evaluate_with_routes(actor)
                    print(f"    seed={seed:2d}  RESTORED  ret={avg_ret:.4f}  fam={dom}")
                    row = {
                        "env_name": ENV_NAME, "dataset_name": ds_name,
                        "algorithm": ALGO, "seed": seed,
                        "avg_return": f"{avg_ret:.4f}",
                        "success_rate": f"{succ_rt:.4f}",
                        "avg_episode_len": f"{avg_len:.2f}",
                        "final_actor_loss": "restored",
                        "final_q_loss": "restored",
                        "final_v_loss": "restored",
                        "checkpoint_path": ckpt_path,
                        "status": "completed",
                    }
                    append_row(SUMMARY_PATH, row, SUMMARY_COLUMNS)
                    route_rows.append({
                        "dataset_name": ds_name, "seed": seed, "dominant_family": dom,
                        "frac_A": f"{fA:.3f}", "frac_B": f"{fB:.3f}",
                        "frac_C": f"{fC:.3f}", "frac_stem": f"{fS:.3f}",
                        "avg_return": f"{avg_ret:.4f}", "note": "restored",
                    })
                    run_count += 1
                    continue
                except Exception:
                    pass  # corrupt — retrain

            # Train
            print(f"    seed={seed:2d}  training...", end=" ", flush=True)
            actor, q1, q2, value_net, al, ql, vl = train_iql(
                obs, acts, rews, nobs, terms, IQL_CFG, seed)
            avg_ret, succ_rt, avg_len, dom, fA, fB, fC, fS = evaluate_with_routes(actor)
            print(f"al={al:.4f} ql={ql:.6f}  ret={avg_ret:.4f}  succ={succ_rt:.3f}  fam={dom}")

            save_iql_ckpt(ckpt_path, actor, q1, q2, value_net)

            row = {
                "env_name": ENV_NAME, "dataset_name": ds_name,
                "algorithm": ALGO, "seed": seed,
                "avg_return":      f"{avg_ret:.4f}",
                "success_rate":    f"{succ_rt:.4f}",
                "avg_episode_len": f"{avg_len:.2f}",
                "final_actor_loss": f"{al:.6f}",
                "final_q_loss":     f"{ql:.6f}",
                "final_v_loss":     f"{vl:.6f}",
                "checkpoint_path": ckpt_path,
                "status": "completed",
            }
            append_row(SUMMARY_PATH, row, SUMMARY_COLUMNS)
            route_rows.append({
                "dataset_name": ds_name, "seed": seed, "dominant_family": dom,
                "frac_A": f"{fA:.3f}", "frac_B": f"{fB:.3f}",
                "frac_C": f"{fC:.3f}", "frac_stem": f"{fS:.3f}",
                "avg_return": f"{avg_ret:.4f}", "note": "trained",
            })
            run_count += 1

    # ── 5. Write route-family CSV ──────────────────────────────────────────
    if route_rows:
        with open(ROUTE_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ROUTE_COLUMNS)
            w.writeheader(); w.writerows(route_rows)
        print(f"\n  Route-family CSV: {ROUTE_PATH}")

    # ── 6. Analysis ────────────────────────────────────────────────────────
    print(f"\n── 6. Analysis ──")
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
        return float(arr.mean() - 1.96*se), float(arr.mean() + 1.96*se)

    sw_mean, sw_std = float(sw_rets.mean()), float(sw_rets.std(ddof=1))
    ln_mean, ln_std = float(ln_rets.mean()), float(ln_rets.std(ddof=1))
    gap = sw_mean - ln_mean
    sw_lo, sw_hi = ci95(sw_rets)
    ln_lo, ln_hi = ci95(ln_rets)
    sw_wins = int(np.sum(sw_rets > ln_rets)) if len(sw_rets)==len(ln_rets) else -1
    ci_no_overlap = sw_lo > ln_hi if not math.isnan(sw_lo) else False

    # Route-family distribution
    route_dist = {}
    for row in route_rows:
        ds = row["dataset_name"]; fam = row["dominant_family"]
        if ds not in route_dist:
            route_dist[ds] = {"A": 0, "B": 0, "C": 0, "fail": 0}
        route_dist[ds][fam] = route_dist[ds].get(fam, 0) + 1

    print(f"  small-wide     n={len(sw_rets)}  mean={sw_mean:.4f}  std={sw_std:.4f}  "
          f"95%CI=[{sw_lo:.4f},{sw_hi:.4f}]")
    print(f"  large-narrow-A n={len(ln_rets)}  mean={ln_mean:.4f}  std={ln_std:.4f}  "
          f"95%CI=[{ln_lo:.4f},{ln_hi:.4f}]")
    print(f"  gap (sw - ln)  = {gap:.4f}")
    for ds, dist in route_dist.items():
        label = "sw" if "wide" in ds else "ln"
        print(f"  route [{label}]: A={dist.get('A',0)} B={dist.get('B',0)} "
              f"C={dist.get('C',0)} fail={dist.get('fail',0)}")

    # Compare with BC
    bc_gap = 0.0200   # from BC formal runlog
    iql_amplifies = gap > bc_gap
    directional = gap > 0
    recommend_cql = directional

    # ── 7. Write RUNLOG ────────────────────────────────────────────────────
    sw_rd = route_dist.get("envB_v2_small_wide", {})
    ln_rd = route_dist.get("envB_v2_large_narrow_A", {})

    lines = [
        "# ENVB_V2_IQL_FORMAL_RUNLOG.md",
        "# EnvB_v2 IQL Formal Validation — 20 Seeds",
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
        "| Environment | EnvB_v2 |",
        "| Algorithm | IQL (Implicit Q-Learning) |",
        "| Seeds | 0–19 (20 per dataset) |",
        f"| Eval episodes | {EVAL_EPISODES} |",
        "| MLP hidden | [256, 256] |",
        f"| Updates | {IQL_CFG['num_updates']} |",
        f"| Batch size | {IQL_CFG['batch_size']} |",
        f"| expectile | {IQL_CFG['expectile']} |",
        f"| temperature | {IQL_CFG['temperature']} |",
        "| small-wide  | 50k, A+B+C, delay=0.25, seed=425 |",
        "| large-narrow-A | 200k, family A, delay=0.10, seed=411 |",
        "| SA coverage gap | 0.2407 − 0.0852 = 0.1556 |",
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
        f"Gap (sw_mean − ln_mean)     = {gap:.4f}",
        f"BC gap (reference)          = {bc_gap:.4f}",
        f"IQL amplifies over BC:       {'YES' if iql_amplifies else 'NO'} (IQL gap={gap:.4f} vs BC gap={bc_gap:.4f})",
    ]
    if sw_wins >= 0:
        lines.append(f"Seed-paired: wide > narrow in {sw_wins}/{len(sw_rets)} seeds")
    lines += [
        f"95% CI overlap:              {'NO (non-overlapping)' if ci_no_overlap else 'YES (overlapping)'}",
        "",
        "## 3. Route-Family Convergence",
        "",
        "IQL greedy policies converge to one of three corridor routes (A/B/C).",
        "The dominant family is inferred from the fraction of corridor-region steps",
        "across all 50 evaluation episodes.",
        "",
        "### small-wide",
        "",
        f"| Family | Count |",
        f"|--------|-------|",
        f"| A (33-step, return≈0.67) | {sw_rd.get('A',0)} |",
        f"| B (21-step, return≈0.79) | {sw_rd.get('B',0)} |",
        f"| C (35-step, return≈0.65) | {sw_rd.get('C',0)} |",
        f"| fail / undetermined | {sw_rd.get('fail',0)} |",
        "",
        "### large-narrow-A",
        "",
        f"| Family | Count |",
        f"|--------|-------|",
        f"| A (33-step, return≈0.67) | {ln_rd.get('A',0)} |",
        f"| B (21-step, return≈0.79) | {ln_rd.get('B',0)} |",
        f"| C (35-step, return≈0.65) | {ln_rd.get('C',0)} |",
        f"| fail / undetermined | {ln_rd.get('fail',0)} |",
        "",
        "**Interpretation:**",
        f"- Narrow-A trains exclusively on corridor A data → expected to converge to A in most seeds",
        f"- Wide trains on A+B+C data → IQL should learn that corridor B has highest Q-value",
        f"  (shortest path = highest discounted return) and redirect policy toward B",
        f"- Difference in B-convergence rate (wide vs narrow) is the coverage effect",
        "",
        "## 4. Directional Assessment",
        "",
    ]

    if directional and ci_no_overlap:
        lines += [
            "**STRONG directional support for wide > narrow.**",
            "",
            f"- Gap = {gap:.4f} (positive)",
            "- 95% CI non-overlapping: effect is statistically clear at 20 seeds",
            f"- IQL {'amplifies' if iql_amplifies else 'does not amplify'} over BC gap ({bc_gap:.4f})",
        ]
    elif directional:
        lines += [
            "**DIRECTIONAL support for wide > narrow (CIs overlap).**",
            "",
            f"- Gap = {gap:.4f} (positive)",
            "- 95% CI overlap: directional but not decisive at n=20",
            f"- IQL {'amplifies' if iql_amplifies else 'matches'} BC gap ({bc_gap:.4f})",
        ]
    else:
        lines += [
            "**NO directional support: wide NOT > narrow at mean level.**",
            "",
            f"- Gap = {gap:.4f} (non-positive)",
            f"- Consistent with 'learning-insensitive under single-goal evaluation' hypothesis",
        ]

    lines += [
        "",
        "## 5. Proceed to CQL Formal 20 Seeds?",
        "",
    ]
    if recommend_cql:
        lines += [
            "**YES — proceed to CQL formal 20 seeds.**",
            "",
            f"Rationale: IQL gap = {gap:.4f} (positive direction).",
            "CQL smoke passed. Running CQL completes the 3-algorithm validation picture.",
        ]
    else:
        lines += [
            "**CONDITIONAL — review before CQL.**",
            "",
            f"IQL gap = {gap:.4f} (non-positive). Before running CQL 20 seeds:",
            "- Confirm whether EnvB_v2 should be classified as learning-insensitive",
            "- CQL adds marginal value if both BC and IQL fail to show directionality",
        ]

    lines += [
        "",
        "## 6. Per-Seed Returns",
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
    print(f"Gap:            {gap:.4f}  (BC was {bc_gap:.4f})")
    print(f"IQL amplifies:  {'YES' if iql_amplifies else 'NO'}")
    print(f"CI non-overlap: {ci_no_overlap}")
    print(f"Proceed to CQL: {'YES' if recommend_cql else 'CONDITIONAL'}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
