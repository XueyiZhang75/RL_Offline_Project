"""
scripts/run_envbc_iql_validation.py
Retrofit Phase R3: IQL EnvB/C retained discrete validation.

4 datasets x 1 algo (IQL) x 20 seeds = 80 runs.
Inherits:
  - IQL_CFG / train_iql / save_iql_checkpoint / load_iql_checkpoint
    from run_envA_v2_iql_sanity.py (R1)
  - encode_envB / encode_envC / encode_single_B / encode_single_C
    / load_validation_dataset / evaluate / check_dataset_schema
    from run_envbc_validation.py (Clean Phase 9)
Supports resumable + append-after-each-run + reuse/verify mode.
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Import R1 IQL infrastructure ─────────────────────────────────────────────
from scripts.run_envA_v2_iql_sanity import (
    IQL_CFG,
    train_iql, save_iql_checkpoint, load_iql_checkpoint, resolve_iql_path,
    AUDIT_PATH, MANIFEST_PATH, PROJECT_ROOT,
)

# ── Import Clean Phase 9 EnvB/C infrastructure ───────────────────────────────
from scripts.run_envbc_validation import (
    encode_envB, encode_envC, encode_single_B, encode_single_C,
    load_validation_dataset, evaluate, check_dataset_schema,
    ENVB_OBS_DIM, ENVC_OBS_DIM, EVAL_EPISODES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

IQL_DIR         = os.path.join(PROJECT_ROOT, "artifacts", "training_iql")
BC_CQL_VAL_SUM  = os.path.join(PROJECT_ROOT, "artifacts", "training_validation",
                                "envbc_validation_summary.csv")
IQL_MAIN_SUM    = os.path.join(PROJECT_ROOT, "artifacts", "training_iql",
                                "envA_v2_iql_main_summary.csv")
SUMMARY_PATH    = os.path.join(IQL_DIR, "envbc_iql_validation_summary.csv")

# ── Frozen constants ──────────────────────────────────────────────────────────

VALIDATION_SEEDS = list(range(20))
IQL_VALIDATION_DATASETS = [
    "envB_small_wide_medium",
    "envB_large_narrow_medium",
    "envC_small_wide_medium",
    "envC_large_narrow_medium",
]
TOTAL_RUNS = len(IQL_VALIDATION_DATASETS) * len(VALIDATION_SEEDS)  # 80

DS_ENV_MAP = {
    "envB_small_wide_medium":    "EnvB",
    "envB_large_narrow_medium":  "EnvB",
    "envC_small_wide_medium":    "EnvC",
    "envC_large_narrow_medium":  "EnvC",
}

REQUIRED_NPZ_KEYS = [
    "observations", "actions", "rewards", "next_observations", "terminals",
    "truncations", "episode_ids", "timesteps",
    "source_policy_ids", "source_train_seeds", "source_behavior_epsilons",
]

_IQL_CKPT_REQUIRED_KEYS = [
    "actor_state_dict", "q1_state_dict", "q2_state_dict", "value_state_dict",
]

SUMMARY_COLUMNS = [
    "dataset_name", "algorithm", "train_seed",
    "num_updates", "final_actor_loss", "final_q_loss", "final_value_loss",
    "eval_episodes", "avg_return", "success_rate", "avg_episode_length",
    "checkpoint_path", "status", "notes",
]


# ── EnvB/C-parametrized IQL training & checkpoint helpers ────────────────────
# train_iql from R1 hardcodes OBS_DIM=900 (EnvA_v2 only).
# These wrappers accept obs_dim explicitly — same algorithm, parametrized input.

def train_iql_envbc(obs, acts, rews, nobs, terms, cfg, seed, obs_dim):
    """Discrete IQL for arbitrary obs_dim (EnvB=225, EnvC=450).
    Identical algorithm to R1 train_iql; obs_dim is parameterized."""
    import torch.nn as nn
    import torch.optim as optim
    from scripts.run_envA_v2_sanity import MLP, N_ACTIONS

    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(obs)

    actor     = MLP(obs_dim, N_ACTIONS, cfg["hidden_dims"])
    q1        = MLP(obs_dim, N_ACTIONS, cfg["hidden_dims"])
    q2        = MLP(obs_dim, N_ACTIONS, cfg["hidden_dims"])
    q1_tgt    = MLP(obs_dim, N_ACTIONS, cfg["hidden_dims"])
    q2_tgt    = MLP(obs_dim, N_ACTIONS, cfg["hidden_dims"])
    value_net = MLP(obs_dim, 1,         cfg["hidden_dims"])

    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(),
                           lr=cfg["actor_lr"],  weight_decay=cfg["weight_decay"])
    q_opt     = optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                           lr=cfg["critic_lr"], weight_decay=cfg["weight_decay"])
    v_opt     = optim.Adam(value_net.parameters(),
                           lr=cfg["value_lr"],  weight_decay=cfg["weight_decay"])

    obs_t   = torch.from_numpy(obs)
    acts_t  = torch.from_numpy(acts)
    rews_t  = torch.from_numpy(rews)
    nobs_t  = torch.from_numpy(nobs)
    terms_t = torch.from_numpy(terms)

    gamma, tau       = cfg["gamma"], cfg["expectile"]
    temperature      = cfg["temperature"]
    adv_clip         = cfg["adv_clip"]
    target_tau       = cfg["target_tau"]

    final_actor_loss = float("nan")
    final_q_loss     = float("nan")
    final_value_loss = float("nan")

    for step in range(cfg["num_updates"]):
        idx  = np.random.randint(0, n, size=cfg["batch_size"])
        s, a, r, ns, done = obs_t[idx], acts_t[idx], rews_t[idx], nobs_t[idx], terms_t[idx]

        # 1. Value (expectile)
        with torch.no_grad():
            q_min_sa = torch.min(
                q1_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1),
                q2_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1))
        v    = value_net(s).squeeze(1)
        diff = q_min_sa - v
        wt   = torch.where(diff >= 0,
                           torch.full_like(diff, tau),
                           torch.full_like(diff, 1.0 - tau))
        value_loss = (wt * diff.pow(2)).mean()
        v_opt.zero_grad(); value_loss.backward(); v_opt.step()
        final_value_loss = value_loss.item()

        # 2. Twin Q
        with torch.no_grad():
            td_target = r + gamma * (1.0 - done) * value_net(ns).squeeze(1)
        q1_a = q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2_a = q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = (nn.functional.mse_loss(q1_a, td_target) +
                  nn.functional.mse_loss(q2_a, td_target))
        q_opt.zero_grad(); q_loss.backward(); q_opt.step()
        final_q_loss = q_loss.item()

        # 3. Advantage-weighted BC actor
        with torch.no_grad():
            adv = (torch.min(q1_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1),
                             q2_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1))
                   - value_net(s).squeeze(1))
            exp_adv = torch.exp(adv / temperature).clamp(max=adv_clip)
        log_probs  = nn.functional.log_softmax(actor(s), dim=1)
        log_pi_a   = log_probs.gather(1, a.unsqueeze(1)).squeeze(1)
        actor_loss = -(exp_adv * log_pi_a).mean()
        actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
        final_actor_loss = actor_loss.item()

        # Soft target update
        for p, tp in zip(q1.parameters(), q1_tgt.parameters()):
            tp.data.copy_(target_tau * p.data + (1.0 - target_tau) * tp.data)
        for p, tp in zip(q2.parameters(), q2_tgt.parameters()):
            tp.data.copy_(target_tau * p.data + (1.0 - target_tau) * tp.data)

    return actor, q1, q2, value_net, final_actor_loss, final_q_loss, final_value_loss


def save_iql_checkpoint_envbc(path, actor, q1, q2, value_net,
                              dataset_name, seed, cfg,
                              actor_loss, q_loss, value_loss, obs_dim):
    """Save IQL checkpoint with actual obs_dim (not hardcoded 900)."""
    from scripts.run_envA_v2_sanity import N_ACTIONS
    torch.save({
        "actor_state_dict":  actor.state_dict(),
        "q1_state_dict":     q1.state_dict(),
        "q2_state_dict":     q2.state_dict(),
        "value_state_dict":  value_net.state_dict(),
        "dataset_name":      dataset_name,
        "train_seed":        seed,
        "num_updates":       cfg["num_updates"],
        "final_actor_loss":  actor_loss,
        "final_q_loss":      q_loss,
        "final_value_loss":  value_loss,
        "obs_dim":           obs_dim,
        "n_actions":         N_ACTIONS,
        "config":            cfg,
    }, path)


# ── Path helper ───────────────────────────────────────────────────────────────

def resolve_val_path(rel_or_abs):
    """Normalize path: handle backslash, relative → absolute."""
    p = rel_or_abs.replace("\\", "/")
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_ROOT, p)
    return os.path.normpath(p)


# ── Pre-flight helpers ────────────────────────────────────────────────────────

def check_bc_cql_val_summary_loadable(summary_path):
    """Verify all 160 BC/CQL validation checkpoints exist and are torch-loadable."""
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 160, f"BC/CQL val rows={len(rows)}, expected 160"
    assert all(r["status"] == "completed" for r in rows), \
        "BC/CQL val: not all status=completed"
    for r in rows:
        cp = r.get("checkpoint_path", "").strip()
        assert cp, f"BC/CQL val: empty checkpoint_path"
        ap = resolve_val_path(cp)
        assert os.path.isfile(ap), f"BC/CQL val ckpt missing: {ap}"
        try:
            torch.load(ap, map_location="cpu", weights_only=False)
        except Exception as e:
            raise AssertionError(f"BC/CQL val ckpt not loadable: {ap} — {e}")


def check_iql_main_summary_valid(summary_path):
    """Verify all 80 IQL main runs are valid (strict run_is_valid criteria)."""
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 80, f"IQL main rows={len(rows)}, expected 80"
    for r in rows:
        assert r["status"] == "completed", \
            f"IQL main run not completed: {r.get('dataset_name')} seed{r.get('train_seed')}"
        cp = r.get("checkpoint_path", "").strip()
        assert cp, f"IQL main: empty checkpoint_path"
        ap = resolve_val_path(cp)
        assert os.path.isfile(ap), f"IQL main ckpt missing: {ap}"
        try:
            ckpt = torch.load(ap, map_location="cpu", weights_only=False)
            for k in _IQL_CKPT_REQUIRED_KEYS:
                assert k in ckpt, f"IQL main ckpt missing key '{k}': {ap}"
        except AssertionError:
            raise
        except Exception as e:
            raise AssertionError(f"IQL main ckpt not loadable: {ap} — {e}")
        try:
            assert math.isfinite(float(r.get("avg_return", "nan")))
            assert math.isfinite(float(r.get("success_rate", "nan")))
        except Exception:
            raise AssertionError(
                f"IQL main non-finite metrics: {r.get('dataset_name')} seed{r.get('train_seed')}")


# ── Run validity ──────────────────────────────────────────────────────────────

def run_is_valid(row):
    """Strict run validity: status + path + existence + loadability + keys + finite metrics."""
    if row.get("status") != "completed":
        return False
    cp = row.get("checkpoint_path", "").strip()
    if not cp:
        return False
    ap = resolve_val_path(cp)
    if not os.path.isfile(ap):
        return False
    try:
        ckpt = torch.load(ap, map_location="cpu", weights_only=False)
        for k in _IQL_CKPT_REQUIRED_KEYS:
            if k not in ckpt:
                return False
    except Exception:
        return False
    try:
        return (math.isfinite(float(row.get("avg_return", "nan"))) and
                math.isfinite(float(row.get("success_rate", "nan"))))
    except (ValueError, TypeError):
        return False


# ── Resume helpers ────────────────────────────────────────────────────────────

def load_completed_runs():
    """Load SUMMARY_PATH → dict keyed (dataset_name, 'iql', str(seed)) → row."""
    if not os.path.isfile(SUMMARY_PATH):
        return {}
    completed = {}
    with open(SUMMARY_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset_name"], row["algorithm"], str(row["train_seed"]))
            completed[key] = row
    return completed


def existing_validation_complete():
    """Return (True, completed_dict) if all 80 runs are strictly valid."""
    completed = load_completed_runs()
    if len(completed) < TOTAL_RUNS:
        return False, completed
    for ds in IQL_VALIDATION_DATASETS:
        for s in VALIDATION_SEEDS:
            if not run_is_valid(completed.get((ds, "iql", str(s)), {})):
                return False, completed
    return True, completed


def append_val_row(row):
    """Append one row to SUMMARY_PATH with fsync."""
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    write_header = not os.path.isfile(SUMMARY_PATH)
    with open(SUMMARY_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ── Frozen-file snapshot helpers ─────────────────────────────────────────────

def _file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_frozen_file_snapshots(paths):
    """Return dict path → {exists, size, sha256} for each path."""
    snap = {}
    for p in paths:
        if os.path.isfile(p):
            snap[p] = {"exists": True, "size": os.path.getsize(p), "sha256": _file_sha256(p)}
        else:
            snap[p] = {"exists": False, "size": None, "sha256": None}
    return snap


def frozen_snapshots_equal(before, after):
    """Return (True, []) if all snapshots match, else (False, [diff_messages])."""
    diffs = []
    for p in before:
        b, a = before[p], after.get(p, {"exists": False})
        if not b["exists"]:
            diffs.append(f"{p}: was missing before snapshot — cannot verify")
            continue
        if not a.get("exists", False):
            diffs.append(f"{p}: existed before but is now missing")
            continue
        if b["size"] != a["size"]:
            diffs.append(f"{p}: size changed {b['size']} → {a['size']}")
        elif b["sha256"] != a["sha256"]:
            diffs.append(f"{p}: content changed (sha256 mismatch)")
    return (len(diffs) == 0), diffs


FROZEN_FILES = [AUDIT_PATH, MANIFEST_PATH, BC_CQL_VAL_SUM, IQL_MAIN_SUM]


# ── Aggregate stats ───────────────────────────────────────────────────────────

def compute_stats(values):
    from scipy import stats as sp_stats
    n = len(values)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n < 2 or s == 0:
        return m, s, (m, m)
    se = s / math.sqrt(n)
    ci = sp_stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return m, s, ci


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Retrofit Phase R3: IQL EnvB/C retained validation")
    print(f"  4 datasets x 1 algo x 20 seeds = {TOTAL_RUNS} runs")
    print("=" * 66)
    print()

    # ── Snapshot frozen files before any logic ────────────────────────────
    _frozen_snapshot_before = capture_frozen_file_snapshots(FROZEN_FILES)

    # ── Pre-flight A: freeze audit ────────────────────────────────────────
    print("-- Pre-flight A: freeze --------------------------------------------")
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 13 and all(r["freeze_ready"] == "yes" for r in audit_rows)
    print("  audit: 13 rows, all freeze_ready=yes")

    # ── Pre-flight B: retained validation datasets schema ─────────────────
    print("-- Pre-flight B: retained validation datasets schema ---------------")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
    for ds in IQL_VALIDATION_DATASETS:
        assert ds in mrows, f"missing from manifest: {ds}"
        p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
        assert os.path.isfile(p), f"npz missing: {p}"
        d = np.load(p, allow_pickle=True)
        for k in REQUIRED_NPZ_KEYS:
            assert k in d, f"{ds} missing key '{k}'"
        n = len(d["observations"])
        env_name = DS_ENV_MAP[ds]
        if env_name == "EnvB":
            assert d["observations"].shape[1] == 2, \
                f"{ds} obs shape {d['observations'].shape}, expected (N,2)"
        elif env_name == "EnvC":
            assert d["observations"].shape[1] == 3, \
                f"{ds} obs shape {d['observations'].shape}, expected (N,3)"
        for k in REQUIRED_NPZ_KEYS[1:]:
            assert len(d[k]) == n, f"{ds} {k} length mismatch"
        print(f"  {ds} ({env_name}): {n} transitions, schema OK")

    # ── Pre-flight C: BC/CQL validation 160/160 loadability ──────────────
    print("-- Pre-flight C: BC/CQL validation summary (full 160 loadability) --")
    check_bc_cql_val_summary_loadable(BC_CQL_VAL_SUM)
    print("  BC/CQL val: 160/160 completed, all ckpts exist and loadable")

    # ── Pre-flight D: R2 IQL main 80/80 loadability ───────────────────────
    print("-- Pre-flight D: R2 IQL main summary (full 80 loadability) ---------")
    check_iql_main_summary_valid(IQL_MAIN_SUM)
    print("  IQL main: 80/80 valid, all ckpts loadable with required keys")
    print()

    # ── Frozen IQL config ─────────────────────────────────────────────────
    print("-- Frozen IQL config (inherited from R1) ---------------------------")
    for k, v in IQL_CFG.items():
        print(f"  {k} = {v}")
    print()

    # ── Decide: reuse / resume / train ────────────────────────────────────
    os.makedirs(IQL_DIR, exist_ok=True)
    complete, completed = existing_validation_complete()

    n_valid = sum(
        1 for ds in IQL_VALIDATION_DATASETS
        for s in VALIDATION_SEEDS
        if run_is_valid(completed.get((ds, "iql", str(s)), {}))
    )

    if complete:
        print(f"-- VERIFY mode: all {TOTAL_RUNS} runs complete, read-only re-verification ---")
    else:
        print(f"-- RESUME/TRAIN mode: {n_valid}/{TOTAL_RUNS} already done, running rest ----")
    print()

    # ── Training / resume loop ────────────────────────────────────────────
    if not complete:
        global_idx = 0
        for ds_name in IQL_VALIDATION_DATASETS:
            env_name = DS_ENV_MAP[ds_name]
            obs, acts, rews, nobs, terms, _, obs_dim = load_validation_dataset(ds_name)
            print(f"-- {ds_name} ({env_name}, {len(obs)} transitions, obs_dim={obs_dim}) --")

            for seed in VALIDATION_SEEDS:
                global_idx += 1
                key = (ds_name, "iql", str(seed))

                if run_is_valid(completed.get(key, {})):
                    print(f"  [{global_idx}/{TOTAL_RUNS}] seed={seed}: SKIP")
                    continue

                tag      = f"{ds_name}_iql_seed{seed}"
                ckpt_abs = os.path.join(IQL_DIR, f"{tag}.pt")
                ckpt_rel = os.path.relpath(ckpt_abs, PROJECT_ROOT).replace("\\", "/")

                print(f"  [{global_idx}/{TOTAL_RUNS}] seed={seed}...", end="", flush=True)
                al = ql = vl = float("nan")
                eval_result = {"avg_return": float("nan"),
                               "success_rate": float("nan"),
                               "avg_episode_length": float("nan")}
                status   = "failed"
                notes    = ""
                ckpt_rel_out = ""

                try:
                    actor, q1, q2, vnet, al, ql, vl = train_iql_envbc(
                        obs, acts, rews, nobs, terms, IQL_CFG, seed, obs_dim)
                    save_iql_checkpoint_envbc(ckpt_abs, actor, q1, q2, vnet,
                                             ds_name, seed, IQL_CFG, al, ql, vl, obs_dim)
                    eval_result  = evaluate(actor, env_name)
                    status       = "completed"
                    notes        = (f"one-hot {obs_dim}-d; discrete IQL EnvB/C validation; "
                                    f"frozen IQL config inherited from R1")
                    ckpt_rel_out = ckpt_rel
                    print(f" al={al:.4f} ql={ql:.4f} vl={vl:.4f}"
                          f" ret={eval_result['avg_return']:.4f}"
                          f" sr={eval_result['success_rate']:.3f}")
                except Exception as e:
                    notes = f"error: {e}"
                    print(f" FAILED: {e}")

                row = {
                    "dataset_name":       ds_name,
                    "algorithm":          "iql",
                    "train_seed":         str(seed),
                    "num_updates":        str(IQL_CFG["num_updates"]),
                    "final_actor_loss":   f"{al:.6f}" if math.isfinite(al) else str(al),
                    "final_q_loss":       f"{ql:.6f}" if math.isfinite(ql) else str(ql),
                    "final_value_loss":   f"{vl:.6f}" if math.isfinite(vl) else str(vl),
                    "eval_episodes":      str(EVAL_EPISODES),
                    "avg_return":         f"{eval_result['avg_return']:.4f}"
                                          if math.isfinite(eval_result["avg_return"])
                                          else str(eval_result["avg_return"]),
                    "success_rate":       f"{eval_result['success_rate']:.4f}"
                                          if math.isfinite(eval_result["success_rate"])
                                          else str(eval_result["success_rate"]),
                    "avg_episode_length": f"{eval_result['avg_episode_length']:.2f}"
                                          if math.isfinite(eval_result["avg_episode_length"])
                                          else str(eval_result["avg_episode_length"]),
                    "checkpoint_path":    ckpt_rel_out,
                    "status":             status,
                    "notes":              notes,
                }
                append_val_row(row)
                completed[key] = row
            print()

        # Finalize: clean sorted rewrite
        clean_rows = [
            completed[(ds, "iql", str(s))]
            for ds in IQL_VALIDATION_DATASETS
            for s in VALIDATION_SEEDS
            if (ds, "iql", str(s)) in completed
        ]
        if len(clean_rows) == TOTAL_RUNS:
            with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
                w.writeheader()
                w.writerows(clean_rows)
            print(f"  Summary CSV finalized: {len(clean_rows)} rows")
        print()

    # ── Reload completed dict ─────────────────────────────────────────────
    completed = load_completed_runs()

    # ── Gate evaluation ───────────────────────────────────────────────────
    print("-- Gate evaluation -------------------------------------------------")
    gate = {}

    gate["exactly_80_rows"]   = len(completed) == TOTAL_RUNS
    gate["all_runs_valid"]    = all(
        run_is_valid(completed.get((ds, "iql", str(s)), {}))
        for ds in IQL_VALIDATION_DATASETS for s in VALIDATION_SEEDS
    )
    gate["all_losses_finite"] = all(
        math.isfinite(float(r.get("final_actor_loss", "nan"))) and
        math.isfinite(float(r.get("final_q_loss", "nan"))) and
        math.isfinite(float(r.get("final_value_loss", "nan")))
        for r in completed.values() if r.get("status") == "completed"
    )
    gate["all_eval_finite"]   = all(
        math.isfinite(float(r.get("avg_return", "nan"))) and
        math.isfinite(float(r.get("success_rate", "nan")))
        for r in completed.values() if r.get("status") == "completed"
    )
    gate["all_groups_20"]     = all(
        sum(1 for s in VALIDATION_SEEDS
            if run_is_valid(completed.get((ds, "iql", str(s)), {}))) == 20
        for ds in IQL_VALIDATION_DATASETS
    )

    # Checkpoint loadability gate
    all_ckpts_ok = True
    ckpt_err = ""
    for ds in IQL_VALIDATION_DATASETS:
        for s in VALIDATION_SEEDS:
            row = completed.get((ds, "iql", str(s)), {})
            cp  = row.get("checkpoint_path", "").strip()
            if not cp:
                all_ckpts_ok = False; ckpt_err = f"{ds} seed{s}: empty path"; break
            ap = resolve_val_path(cp)
            if not os.path.isfile(ap):
                all_ckpts_ok = False; ckpt_err = f"missing: {ap}"; break
            try:
                ckpt = torch.load(ap, map_location="cpu", weights_only=False)
                for k in _IQL_CKPT_REQUIRED_KEYS:
                    if k not in ckpt:
                        raise KeyError(f"missing '{k}'")
            except Exception as e:
                all_ckpts_ok = False; ckpt_err = f"{ap}: {e}"; break
        if not all_ckpts_ok:
            break
    gate["all_ckpts_loadable"] = all_ckpts_ok
    if not all_ckpts_ok:
        print(f"  CKPT LOAD FAIL: {ckpt_err}")

    _frozen_snapshot_after = capture_frozen_file_snapshots(FROZEN_FILES)
    _frozen_ok, _frozen_diffs = frozen_snapshots_equal(_frozen_snapshot_before,
                                                        _frozen_snapshot_after)
    gate["no_frozen_files_modified"] = _frozen_ok
    if not _frozen_ok:
        for diff in _frozen_diffs:
            print(f"  FROZEN FILE MODIFIED: {diff}")

    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")

    go = all(gate.values())
    print()

    # ── Aggregate results ─────────────────────────────────────────────────
    if go:
        print("-- Aggregated IQL EnvB/C validation results (20-seed) --------------")
        agg = {}
        for ds in IQL_VALIDATION_DATASETS:
            rets = [float(completed[(ds, "iql", str(s))]["avg_return"])
                    for s in VALIDATION_SEEDS]
            srs  = [float(completed[(ds, "iql", str(s))]["success_rate"])
                    for s in VALIDATION_SEEDS]
            rm, rs, rci = compute_stats(rets)
            sm, ss, sci = compute_stats(srs)
            agg[ds] = {"ret_m": rm}
            env_name = DS_ENV_MAP[ds]
            print(f"  {ds} ({env_name}):")
            print(f"    return:  mean={rm:.4f}  std={rs:.4f}  "
                  f"95%CI=[{rci[0]:.4f}, {rci[1]:.4f}]")
            print(f"    success: mean={sm:.4f}  std={ss:.4f}  "
                  f"95%CI=[{sci[0]:.4f}, {sci[1]:.4f}]")

        print()
        print("-- Coverage contrasts (wide - narrow, for record only) -------------")
        for env_prefix in ["envB", "envC"]:
            wide_ds   = f"{env_prefix}_small_wide_medium"
            narrow_ds = f"{env_prefix}_large_narrow_medium"
            diff = agg[wide_ds]["ret_m"] - agg[narrow_ds]["ret_m"]
            print(f"  {env_prefix}: small_wide - large_narrow = {diff:+.4f}  (record only, not a gate)")
        print()

    if go:
        print("Retrofit Phase R3: PASS")
    else:
        failed = [k for k, v in gate.items() if not v]
        print(f"Retrofit Phase R3: FAIL — {failed}")
        sys.exit(1)
