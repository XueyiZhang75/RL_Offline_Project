"""
scripts/run_envbc_validation.py
Clean Phase 9: EnvB / EnvC retained discrete validation.
4 datasets x 2 algos x 20 seeds = 160 runs.

Reuses frozen BC_CFG / CQL_CFG values from sanity. Adapts input_dim per env.
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_envA_v2_sanity import (
    BC_CFG, CQL_CFG, EVAL_EPISODES, N_ACTIONS,
    MLP, SUMMARY_COLUMNS as _BASE_COLS,
    AUDIT_PATH,
)
from scripts.run_envA_v2_main_experiment import (
    MAIN_SUMMARY as ENVA_MAIN_SUMMARY,
    resolve_ckpt_path, check_main_checkpoints_loadable,
)
from envs.gridworld_envs import EnvB, EnvC, HORIZON

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "artifacts", "final_datasets",
                             "final_dataset_manifest.csv")
VAL_DIR    = os.path.join(PROJECT_ROOT, "artifacts", "training_validation")
VAL_SUMMARY = os.path.join(VAL_DIR, "envbc_validation_summary.csv")

# ── Frozen constants ──────────────────────────────────────────────────────────

VALIDATION_SEEDS = list(range(20))
VALIDATION_DATASETS = [
    "envB_small_wide_medium",
    "envB_large_narrow_medium",
    "envC_small_wide_medium",
    "envC_large_narrow_medium",
]
ALGORITHMS = ["bc", "cql"]

ENVB_OBS_DIM = 15 * 15        # 225
ENVC_OBS_DIM = 15 * 15 * 2    # 450

SUMMARY_COLUMNS = [
    "env_name", "dataset_name", "algorithm", "train_seed", "obs_dim",
    "num_updates", "final_train_loss",
    "eval_episodes", "avg_return", "success_rate", "avg_episode_length",
    "checkpoint_path", "status", "notes",
]

REQUIRED_KEYS = {
    "observations", "actions", "rewards", "next_observations",
    "terminals", "truncations", "episode_ids", "timesteps",
    "source_policy_ids", "source_train_seeds", "source_behavior_epsilons",
}

# ── Path normalization helper ─────────────────────────────────────────────────

def resolve_val_path(rel_path):
    """Normalize a checkpoint_path from summary CSV to an absolute path."""
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel_path.replace("\\", "/")))


# ── Pre-flight helpers ────────────────────────────────────────────────────────

def check_main_ckpts_loadable(main_rows):
    """Check all 160 EnvA_v2 main checkpoints are loadable. Returns (ok, failed)."""
    failed = []
    for r in main_rows:
        cp = r.get("checkpoint_path", "").strip()
        if not cp:
            failed.append(f"empty path for {r.get('dataset_name','?')}")
            continue
        abs_path = resolve_val_path(cp)
        try:
            torch.load(abs_path, weights_only=False)
        except Exception as e:
            failed.append(f"{cp}: {e}")
    return len(failed) == 0, failed


def check_dataset_schema(dataset_name, env_name):
    """Check raw .npz schema: required keys, length consistency, obs shape.
    Returns (ok, notes_list)."""
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fpath = None
    for r in rows:
        if r["dataset_name"] == dataset_name:
            fpath = os.path.join(PROJECT_ROOT, r["file_path"])
            break
    if fpath is None:
        return False, [f"{dataset_name} not in manifest"]
    notes = []
    try:
        d = np.load(fpath, allow_pickle=True)
    except Exception as e:
        return False, [f"load error: {e}"]
    keys = set(d.files)
    missing = REQUIRED_KEYS - keys
    if missing:
        notes.append(f"missing keys: {missing}")
    n = len(d["observations"]) if "observations" in keys else 0
    for k in REQUIRED_KEYS:
        if k in keys and len(d[k]) != n:
            notes.append(f"length mismatch: {k}")
    if "observations" in keys:
        obs_shape = d["observations"].shape
        if env_name == "EnvB":
            if len(obs_shape) != 2 or obs_shape[1] != 2:
                notes.append(f"EnvB obs shape {obs_shape}, expected (N,2)")
        elif env_name == "EnvC":
            if len(obs_shape) != 2 or obs_shape[1] != 3:
                notes.append(f"EnvC obs shape {obs_shape}, expected (N,3)")
    return len(notes) == 0, notes


def existing_val_summary_is_complete():
    """Check if VAL_SUMMARY already has 160 completed rows with non-empty checkpoint_paths."""
    if not os.path.isfile(VAL_SUMMARY):
        return False, []
    with open(VAL_SUMMARY, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 160:
        return False, rows
    if not all(r["status"] == "completed" for r in rows):
        return False, rows
    if not all(r.get("checkpoint_path", "").strip() for r in rows):
        return False, rows
    return True, rows


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_envB(obs_array):
    """(N,2) int32 -> (N,225) float32 one-hot."""
    n = len(obs_array)
    enc = np.zeros((n, ENVB_OBS_DIM), dtype=np.float32)
    for i in range(n):
        r, c = int(obs_array[i, 0]), int(obs_array[i, 1])
        enc[i, r * 15 + c] = 1.0
    return enc


def encode_envC(obs_array):
    """(N,3) int32 [row,col,has_key] -> (N,450) float32 one-hot."""
    n = len(obs_array)
    enc = np.zeros((n, ENVC_OBS_DIM), dtype=np.float32)
    for i in range(n):
        r, c, hk = int(obs_array[i, 0]), int(obs_array[i, 1]), int(obs_array[i, 2])
        enc[i, hk * 225 + r * 15 + c] = 1.0
    return enc


def encode_single_B(r, c):
    t = torch.zeros(ENVB_OBS_DIM, dtype=torch.float32)
    t[r * 15 + c] = 1.0
    return t


def encode_single_C(state):
    """state = ((row,col), has_key)"""
    (r, c), hk = state
    t = torch.zeros(ENVC_OBS_DIM, dtype=torch.float32)
    t[hk * 225 + r * 15 + c] = 1.0
    return t


# ── Data loading ──────────────────────────────────────────────────────────────

def load_validation_dataset(dataset_name):
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fpath = None
    env_name = None
    for r in rows:
        if r["dataset_name"] == dataset_name:
            fpath = os.path.join(PROJECT_ROOT, r["file_path"])
            env_name = r["env_name"]
            break
    assert fpath is not None, f"{dataset_name} not in manifest"
    d = np.load(fpath, allow_pickle=True)
    if env_name == "EnvB":
        obs = encode_envB(d["observations"])
        nobs = encode_envB(d["next_observations"])
        obs_dim = ENVB_OBS_DIM
    elif env_name == "EnvC":
        obs = encode_envC(d["observations"])
        nobs = encode_envC(d["next_observations"])
        obs_dim = ENVC_OBS_DIM
    else:
        raise ValueError(f"Unexpected env: {env_name}")
    acts = d["actions"].astype(np.int64)
    rews = d["rewards"].astype(np.float32)
    terms = d["terminals"].astype(np.float32)
    return obs, acts, rews, nobs, terms, env_name, obs_dim


# ── Training (parametrized by obs_dim) ────────────────────────────────────────

def train_bc(obs, acts, cfg, seed, obs_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(obs)
    model = MLP(obs_dim, N_ACTIONS, cfg["hidden_dims"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    obs_t = torch.from_numpy(obs)
    acts_t = torch.from_numpy(acts)
    final_loss = float("nan")
    for step in range(cfg["num_updates"]):
        idx = np.random.randint(0, n, size=cfg["batch_size"])
        logits = model(obs_t[idx])
        loss = loss_fn(logits, acts_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    return model, final_loss


def train_cql(obs, acts, rews, nobs, terms, cfg, seed, obs_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(obs)
    q_net = MLP(obs_dim, N_ACTIONS, cfg["hidden_dims"])
    target_net = MLP(obs_dim, N_ACTIONS, cfg["hidden_dims"])
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    obs_t = torch.from_numpy(obs)
    acts_t = torch.from_numpy(acts)
    rews_t = torch.from_numpy(rews)
    nobs_t = torch.from_numpy(nobs)
    terms_t = torch.from_numpy(terms)
    gamma, alpha = cfg["gamma"], cfg["cql_alpha"]
    final_loss = float("nan")
    for step in range(cfg["num_updates"]):
        idx = np.random.randint(0, n, size=cfg["batch_size"])
        s, a, r, ns, done = obs_t[idx], acts_t[idx], rews_t[idx], nobs_t[idx], terms_t[idx]
        with torch.no_grad():
            max_q_next = target_net(ns).max(dim=1).values
            td_target = r + gamma * (1 - done) * max_q_next
        q_vals = q_net(s)
        q_a = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)
        td_loss = nn.functional.mse_loss(q_a, td_target)
        cql_penalty = (torch.logsumexp(q_vals, dim=1) - q_a).mean()
        loss = td_loss + alpha * cql_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if (step + 1) % 100 == 0:
            for p, tp in zip(q_net.parameters(), target_net.parameters()):
                tp.data.copy_(0.995 * tp.data + 0.005 * p.data)
    return q_net, final_loss


# ── Online evaluation ─────────────────────────────────────────────────────────

def evaluate(model, env_name, n_episodes=EVAL_EPISODES):
    if env_name == "EnvB":
        env = EnvB()
        enc_fn = lambda obs: encode_single_B(obs[0], obs[1])
    elif env_name == "EnvC":
        env = EnvC()
        enc_fn = lambda obs: encode_single_C(obs)
    else:
        raise ValueError(env_name)
    returns, successes, lengths = [], [], []
    model.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done, ep_ret, ep_len, success = False, 0.0, 0, False
            while not done:
                s_t = enc_fn(obs).unsqueeze(0)
                action = int(model(s_t).argmax(dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_ret += reward
                ep_len += 1
                done = terminated or truncated
                if terminated:
                    success = True
            returns.append(ep_ret)
            successes.append(float(success))
            lengths.append(ep_len)
    model.train()
    return {
        "avg_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "avg_episode_length": float(np.mean(lengths)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Clean Phase 9: EnvB / EnvC retained discrete validation")
    print(f"  4 datasets x 2 algos x 20 seeds = 160 runs")
    print("=" * 66)
    print()

    # ── Pre-flight A: freeze ──────────────────────────────────────────────
    print("-- Pre-flight A: freeze --------------------------------------------")
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 13
    for ar in audit_rows:
        assert ar["freeze_ready"] == "yes"
    print("  freeze_ready = yes for all 13: OK")

    # ── Pre-flight B: main experiment (existence + loadability) ──────────
    print("-- Pre-flight B: EnvA_v2 main experiment ----------------------------")
    with open(ENVA_MAIN_SUMMARY, newline="", encoding="utf-8") as f:
        main_rows = list(csv.DictReader(f))
    assert len(main_rows) == 160, f"Main summary rows {len(main_rows)} != 160"
    assert all(r["status"] == "completed" for r in main_rows), "Not all main runs completed"
    for r in main_rows:
        cp = r.get("checkpoint_path", "").strip()
        assert cp and os.path.isfile(resolve_val_path(cp)), f"Missing: {cp}"
    print("  160/160 main runs completed, checkpoints exist: OK")
    main_load_ok, main_load_fail = check_main_ckpts_loadable(main_rows)
    if not main_load_ok:
        print(f"  FAIL: main checkpoints not loadable: {main_load_fail[:3]}")
        sys.exit(1)
    print("  160/160 main checkpoints loadable: OK")

    # ── Pre-flight C: validation dataset schema ───────────────────────────
    print("-- Pre-flight C: validation dataset schema --------------------------")
    ds_env_map = {
        "envB_small_wide_medium": "EnvB", "envB_large_narrow_medium": "EnvB",
        "envC_small_wide_medium": "EnvC", "envC_large_narrow_medium": "EnvC",
    }
    for dn in VALIDATION_DATASETS:
        schema_ok, schema_notes = check_dataset_schema(dn, ds_env_map[dn])
        if not schema_ok:
            print(f"  FAIL: {dn} schema: {schema_notes}")
            sys.exit(1)
        print(f"  {dn}: schema OK")

    # ── Pre-flight D: load encoded validation datasets ────────────────────
    print("-- Pre-flight D: load encoded datasets ------------------------------")
    for dn in VALIDATION_DATASETS:
        obs, acts, rews, nobs, terms, env_name, obs_dim = load_validation_dataset(dn)
        print(f"  {dn}: {len(obs)} trans, env={env_name}, obs_dim={obs_dim}, loaded OK")
    print()

    # ── Print frozen configs ──────────────────────────────────────────────
    print("-- Frozen configs (inherited from sanity) ---------------------------")
    print(f"  BC_CFG:  {BC_CFG}")
    print(f"  CQL_CFG: {CQL_CFG}")
    print(f"  VALIDATION_SEEDS: {VALIDATION_SEEDS[0]}..{VALIDATION_SEEDS[-1]} ({len(VALIDATION_SEEDS)} seeds)")
    print()

    # ── Decide: reuse existing or train ───────────────────────────────────
    complete, existing_rows = existing_val_summary_is_complete()

    if complete:
        print("-- Reuse mode: existing summary is complete (160 rows) -------------")
        print("  Skipping training. Running read-only re-verification.")
        summary_rows = existing_rows
    else:
        print("-- Training mode: running 160 validation runs -----------------------")
        os.makedirs(VAL_DIR, exist_ok=True)
        summary_rows = []
        run_count = 0
        total_runs = len(VALIDATION_DATASETS) * len(ALGORITHMS) * len(VALIDATION_SEEDS)

        for dn in VALIDATION_DATASETS:
            obs, acts, rews, nobs, terms, env_name, obs_dim = load_validation_dataset(dn)
            print(f"-- {dn} ({len(obs)} trans, obs_dim={obs_dim}) --")

            for algo in ALGORITHMS:
                for seed in VALIDATION_SEEDS:
                    run_count += 1
                    tag = f"{dn}_{algo}_seed{seed}"
                    print(f"  [{run_count}/{total_runs}] {tag}...", end="", flush=True)

                    try:
                        if algo == "bc":
                            model, final_loss = train_bc(obs, acts, BC_CFG, seed, obs_dim)
                            num_updates = BC_CFG["num_updates"]
                        else:
                            model, final_loss = train_cql(
                                obs, acts, rews, nobs, terms, CQL_CFG, seed, obs_dim)
                            num_updates = CQL_CFG["num_updates"]

                        ckpt_name = f"{tag}.pt"
                        ckpt_path = os.path.join(VAL_DIR, ckpt_name)
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "algorithm": algo,
                            "env_name": env_name,
                            "dataset_name": dn,
                            "train_seed": seed,
                            "obs_dim": obs_dim,
                            "num_updates": num_updates,
                            "final_train_loss": final_loss,
                            "n_actions": N_ACTIONS,
                            "config": BC_CFG if algo == "bc" else CQL_CFG,
                        }, ckpt_path)

                        ev = evaluate(model, env_name)
                        status = "completed"
                        notes = (f"one-hot {obs_dim}-d; retained validation; "
                                 f"frozen {'BC_CFG' if algo == 'bc' else 'CQL_CFG'} "
                                 "inherited from sanity")
                        print(f" loss={final_loss:.4f}  SR={ev['success_rate']:.3f}  "
                              f"ret={ev['avg_return']:.3f}")

                    except Exception as e:
                        final_loss = float("nan")
                        ev = {"avg_return": float("nan"), "success_rate": float("nan"),
                              "avg_episode_length": float("nan")}
                        ckpt_path = ""
                        num_updates = 0
                        status = "failed"
                        notes = f"error: {e}"
                        print(f" FAILED: {e}")

                    summary_rows.append({
                        "env_name": env_name,
                        "dataset_name": dn,
                        "algorithm": algo,
                        "train_seed": seed,
                        "obs_dim": obs_dim,
                        "num_updates": num_updates,
                        "final_train_loss": f"{final_loss:.6f}" if math.isfinite(final_loss) else str(final_loss),
                        "eval_episodes": EVAL_EPISODES,
                        "avg_return": f"{ev['avg_return']:.4f}" if math.isfinite(ev['avg_return']) else str(ev['avg_return']),
                        "success_rate": f"{ev['success_rate']:.4f}" if math.isfinite(ev['success_rate']) else str(ev['success_rate']),
                        "avg_episode_length": f"{ev['avg_episode_length']:.2f}" if math.isfinite(ev['avg_episode_length']) else str(ev['avg_episode_length']),
                        "checkpoint_path": os.path.relpath(ckpt_path, PROJECT_ROOT) if ckpt_path else "",
                        "status": status,
                        "notes": notes,
                    })

        with open(VAL_SUMMARY, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(summary_rows)
        print()
        print(f"  Validation summary: {VAL_SUMMARY} ({len(summary_rows)} rows)")

    # ── Gate evaluation ───────────────────────────────────────────────────
    print()
    print("-- Gate evaluation -------------------------------------------------")
    gate = {}

    all_completed = all(r["status"] == "completed" for r in summary_rows)
    gate["all_160_completed"] = all_completed
    print(f"  [{'OK' if all_completed else 'FAIL'}] 160/160 completed")

    all_ckpts_exist = all(
        r.get("checkpoint_path", "").strip() and
        os.path.isfile(resolve_val_path(r["checkpoint_path"]))
        for r in summary_rows
    )
    gate["all_160_ckpts_exist"] = all_ckpts_exist
    print(f"  [{'OK' if all_ckpts_exist else 'FAIL'}] 160/160 checkpoints exist")

    all_loadable = True
    for r in summary_rows:
        cp = r.get("checkpoint_path", "").strip()
        if not cp:
            all_loadable = False
            break
        try:
            torch.load(resolve_val_path(cp), weights_only=False)
        except:
            all_loadable = False
            break
    gate["all_160_ckpts_loadable"] = all_loadable
    print(f"  [{'OK' if all_loadable else 'FAIL'}] 160/160 checkpoints loadable")

    all_loss_finite = all(math.isfinite(float(r["final_train_loss"])) for r in summary_rows)
    gate["all_losses_finite"] = all_loss_finite
    print(f"  [{'OK' if all_loss_finite else 'FAIL'}] all losses finite")

    all_eval_finite = all(
        math.isfinite(float(r["avg_return"])) and math.isfinite(float(r["success_rate"]))
        for r in summary_rows
    )
    gate["all_eval_finite"] = all_eval_finite
    print(f"  [{'OK' if all_eval_finite else 'FAIL'}] all eval finite")

    correct_160 = len(summary_rows) == 160
    gate["exactly_160_rows"] = correct_160
    print(f"  [{'OK' if correct_160 else 'FAIL'}] summary has 160 rows")

    all_20 = True
    for dn in VALIDATION_DATASETS:
        for algo in ALGORITHMS:
            cnt = sum(1 for r in summary_rows
                      if r["dataset_name"] == dn and r["algorithm"] == algo
                      and r["status"] == "completed")
            if cnt != 20:
                all_20 = False
    gate["all_groups_20"] = all_20
    print(f"  [{'OK' if all_20 else 'FAIL'}] each group has 20 runs")

    print()
    go = all(gate.values())
    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print()

    # ── Aggregated results ────────────────────────────────────────────────
    if go:
        print("-- Aggregated results (20-seed) ------------------------------------")
        from scipy import stats as sp_stats

        agg = {}
        for dn in VALIDATION_DATASETS:
            for algo in ALGORITHMS:
                rets = [float(r["avg_return"]) for r in summary_rows
                        if r["dataset_name"] == dn and r["algorithm"] == algo]
                srs = [float(r["success_rate"]) for r in summary_rows
                       if r["dataset_name"] == dn and r["algorithm"] == algo]

                ret_mean = np.mean(rets)
                ret_std = np.std(rets, ddof=1)
                ret_se = ret_std / math.sqrt(len(rets)) if ret_std > 0 else 0
                ret_ci = (sp_stats.t.interval(0.95, df=len(rets)-1, loc=ret_mean, scale=ret_se)
                          if ret_se > 0 else (ret_mean, ret_mean))

                sr_mean = np.mean(srs)
                sr_std = np.std(srs, ddof=1)
                sr_se = sr_std / math.sqrt(len(srs)) if sr_std > 0 else 0
                sr_ci = (sp_stats.t.interval(0.95, df=len(srs)-1, loc=sr_mean, scale=sr_se)
                         if sr_se > 0 else (sr_mean, sr_mean))

                agg[(dn, algo)] = {"ret_mean": ret_mean, "sr_mean": sr_mean}
                print(f"  {dn} x {algo}:")
                print(f"    return: mean={ret_mean:.4f}  std={ret_std:.4f}  "
                      f"95%CI=[{ret_ci[0]:.4f}, {ret_ci[1]:.4f}]")
                print(f"    SR:     mean={sr_mean:.4f}  std={sr_std:.4f}  "
                      f"95%CI=[{sr_ci[0]:.4f}, {sr_ci[1]:.4f}]")

        print()
        print("-- small_wide - large_narrow mean return differences ----------------")
        for env_prefix in ["envB", "envC"]:
            sw = f"{env_prefix}_small_wide_medium"
            ln = f"{env_prefix}_large_narrow_medium"
            for algo in ALGORITHMS:
                diff = agg[(sw, algo)]["ret_mean"] - agg[(ln, algo)]["ret_mean"]
                print(f"  {env_prefix} / {algo}: {diff:+.4f}")
        print()

    if go:
        print("Clean Phase 9: PASS")
    else:
        failed = [k for k, v in gate.items() if not v]
        print("Clean Phase 9: FAIL")
        print(f"  Failed: {failed}")
        sys.exit(1)
