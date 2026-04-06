"""
scripts/run_hopper_benchmark.py
Clean Phase 12 重执行版: Hopper minimum benchmark.
3 datasets x 4 algos x 5 seeds = 60 runs.

Execution design:
  - Resumable     : skip completed runs based on seed CSV
  - Append-per-run: CSV append + fsync after every run (crash-safe durability)
  - No log bloat  : NoopAdapterFactory + per-run scratch cleanup
  - Sequential    : one run at a time, gc.collect() + del between runs
  - Verify mode   : if all 60 complete, read-only re-verification only
"""

import sys, os, csv, math, shutil, logging, gc
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("d3rlpy").setLevel(logging.WARNING)
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import numpy as np
import h5py
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from d3rlpy.logging import NoopAdapterFactory as _NoopAdapterFactory
    _NOOP_LOGGER = _NoopAdapterFactory()
except ImportError:
    _NOOP_LOGGER = None

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_PATH        = os.path.join(PROJECT_ROOT, "artifacts", "final_datasets",   "final_dataset_audit.csv")
MAIN_SUMMARY      = os.path.join(PROJECT_ROOT, "artifacts", "training_main",     "envA_v2_main_summary.csv")
VAL_SUMMARY       = os.path.join(PROJECT_ROOT, "artifacts", "training_validation","envbc_validation_summary.csv")
QUALITY_SUMMARY   = os.path.join(PROJECT_ROOT, "artifacts", "training_quality",  "envA_v2_quality_summary.csv")
MECHANISM_SUMMARY = os.path.join(PROJECT_ROOT, "artifacts", "analysis",          "envA_v2_mechanism_summary.csv")
BENCH_DIR         = os.path.join(PROJECT_ROOT, "artifacts", "training_benchmark")
SEED_CSV          = os.path.join(BENCH_DIR, "hopper_benchmark_seed_results.csv")
SUMMARY_CSV       = os.path.join(BENCH_DIR, "hopper_benchmark_summary.csv")
DATA_CACHE_DIR    = os.path.join(BENCH_DIR, "d4rl_cache")
SCRATCH_DIR       = os.path.join(BENCH_DIR, "_backend_tmp")

# ── Frozen benchmark scope ────────────────────────────────────────────────────

BENCHMARK_SEEDS         = list(range(5))   # reduced for feasibility; scope otherwise frozen
BENCHMARK_EVAL_EPISODES = 20
ALGO_N_STEPS            = 100_000

BENCHMARK_DATASETS = {
    "hopper-medium": {
        "url":    "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
        "env_id": "Hopper-v4",
    },
    "hopper-medium-replay": {
        "url":    "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
        "env_id": "Hopper-v4",
    },
    "hopper-medium-expert": {
        "url":    "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_expert-v2.hdf5",
        "env_id": "Hopper-v4",
    },
}

ALGORITHMS = ["bc", "cql", "iql", "td3bc"]

# Frozen algorithm configs (d3rlpy constructor kwargs — same for all 3 datasets)
BC_CFG    = {"batch_size": 100}
CQL_CFG   = {"batch_size": 256}
IQL_CFG   = {"batch_size": 256}
TD3BC_CFG = {"batch_size": 256}

# D4RL Hopper normalization constants (from D4RL paper)
HOPPER_RANDOM_SCORE = 0.0
HOPPER_EXPERT_SCORE = 3234.3

TOTAL_RUNS = len(BENCHMARK_DATASETS) * len(ALGORITHMS) * len(BENCHMARK_SEEDS)  # 60

SEED_COLUMNS = [
    "dataset_name", "resolved_env_id", "algorithm", "train_seed",
    "num_updates_or_epochs", "eval_episodes",
    "raw_return", "normalized_score",
    "checkpoint_path", "status", "notes",
]

SUMMARY_COLUMNS = [
    "dataset_name", "resolved_env_id", "algorithm", "n_runs",
    "mean_raw_return", "std_raw_return",
    "ci95_raw_return_low", "ci95_raw_return_high",
    "mean_normalized_score", "std_normalized_score",
    "ci95_normalized_score_low", "ci95_normalized_score_high",
    "notes",
]


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_path(rel_or_abs):
    """Resolve possibly-relative, possibly-backslashed path to absolute."""
    p = rel_or_abs.replace("\\", "/")
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_ROOT, p)
    return os.path.normpath(p)


def normalize_ckpt_path(abs_path):
    """Convert absolute checkpoint path to PROJECT_ROOT-relative, forward-slash."""
    rel = os.path.relpath(abs_path, PROJECT_ROOT)
    return rel.replace("\\", "/")


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_score(raw):
    return (raw - HOPPER_RANDOM_SCORE) / (HOPPER_EXPERT_SCORE - HOPPER_RANDOM_SCORE) * 100.0


# ── Scratch / log cleanup ─────────────────────────────────────────────────────

def cleanup_backend_scratch():
    """Remove backend scratch / intermediate log dirs after each run."""
    for candidate in [
        SCRATCH_DIR,
        os.path.join(os.getcwd(), "d3rlpy_logs"),
        os.path.join(PROJECT_ROOT, "d3rlpy_logs"),
    ]:
        if os.path.isdir(candidate):
            shutil.rmtree(candidate, ignore_errors=True)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def download_dataset(ds_name):
    """Download D4RL HDF5 if not cached. Returns local path."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    fname = ds_name.replace("-", "_") + ".hdf5"
    local = os.path.join(DATA_CACHE_DIR, fname)
    if not os.path.exists(local):
        url = BENCHMARK_DATASETS[ds_name]["url"]
        print(f"    Downloading {ds_name}...")
        urllib.request.urlretrieve(url, local)
    return local


def load_d4rl_dataset(hdf5_path):
    """Load HDF5 into d3rlpy MDPDataset."""
    import d3rlpy
    with h5py.File(hdf5_path, "r") as f:
        obs   = np.array(f["observations"], dtype=np.float32)
        acts  = np.array(f["actions"],      dtype=np.float32)
        rews  = np.array(f["rewards"],      dtype=np.float32)
        terms = np.array(f["terminals"],    dtype=bool)
        tos   = np.array(f["timeouts"],     dtype=bool)
    return d3rlpy.dataset.MDPDataset(
        observations=obs, actions=acts, rewards=rews,
        terminals=terms | tos,
    )


# ── Algorithm helpers ─────────────────────────────────────────────────────────

def create_algo(algo_name, seed):
    """Create d3rlpy algorithm instance with frozen config."""
    import d3rlpy
    np.random.seed(seed)
    if   algo_name == "bc":    return d3rlpy.algos.BCConfig(**BC_CFG).create(device="cpu:0")
    elif algo_name == "cql":   return d3rlpy.algos.CQLConfig(**CQL_CFG).create(device="cpu:0")
    elif algo_name == "iql":   return d3rlpy.algos.IQLConfig(**IQL_CFG).create(device="cpu:0")
    elif algo_name == "td3bc": return d3rlpy.algos.TD3PlusBCConfig(**TD3BC_CFG).create(device="cpu:0")
    raise ValueError(f"Unknown algo: {algo_name}")


def evaluate_policy(algo_obj, env_id, n_episodes):
    """Evaluate d3rlpy policy. Returns list of episode returns."""
    import gymnasium as gym
    env = gym.make(env_id)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            action = algo_obj.predict(np.expand_dims(obs, 0))[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    env.close()
    return returns


# ── Resume management ─────────────────────────────────────────────────────────

def run_is_valid(row):
    """Return True iff this seed row represents a fully valid completed run."""
    if row.get("status") != "completed":
        return False
    cp = row.get("checkpoint_path", "").strip()
    if not cp:
        return False
    abs_cp = resolve_path(cp)
    if not (os.path.isfile(abs_cp) or os.path.isdir(abs_cp)):
        return False
    try:
        return math.isfinite(float(row.get("normalized_score", "nan")))
    except (ValueError, TypeError):
        return False


def load_completed_runs():
    """Load SEED_CSV → dict keyed (dataset, algo, str(seed)) → row.
    If duplicate keys exist, last row wins (most recent attempt)."""
    if not os.path.isfile(SEED_CSV):
        return {}
    completed = {}
    with open(SEED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset_name"], row["algorithm"], str(row["train_seed"]))
            completed[key] = row
    return completed


def append_seed_row(row):
    """Append one completed/failed row to SEED_CSV with fsync (crash-safe)."""
    os.makedirs(os.path.dirname(SEED_CSV), exist_ok=True)
    write_header = not os.path.isfile(SEED_CSV)
    with open(SEED_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SEED_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def existing_benchmark_complete():
    """Return (True, rows) if all TOTAL_RUNS are valid, else (False, rows)."""
    completed = load_completed_runs()
    all_rows = list(completed.values())
    if len(completed) < TOTAL_RUNS:
        return False, all_rows
    for ds in BENCHMARK_DATASETS:
        for algo in ALGORITHMS:
            for s in BENCHMARK_SEEDS:
                if not run_is_valid(completed.get((ds, algo, str(s)), {})):
                    return False, all_rows
    return True, all_rows


# ── Summary rebuild ───────────────────────────────────────────────────────────

def rebuild_summary(completed_dict):
    """Build 12-row aggregate summary from completed_dict."""
    from scipy import stats as sp_stats
    rows = []
    for ds in BENCHMARK_DATASETS:
        env_id = BENCHMARK_DATASETS[ds]["env_id"]
        for algo in ALGORITHMS:
            group = [
                completed_dict[(ds, algo, str(s))]
                for s in BENCHMARK_SEEDS
                if (ds, algo, str(s)) in completed_dict
                and run_is_valid(completed_dict[(ds, algo, str(s))])
            ]
            if not group:
                continue
            raws  = [float(r["raw_return"])      for r in group]
            norms = [float(r["normalized_score"]) for r in group]
            n = len(raws)

            def _ci95(vals):
                m = np.mean(vals)
                s = np.std(vals, ddof=1) if n > 1 else 0.0
                if n < 2 or s == 0:
                    return m, m
                se = s / math.sqrt(n)
                lo, hi = sp_stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
                return lo, hi

            raw_ci  = _ci95(raws)
            norm_ci = _ci95(norms)
            raw_m,  raw_s  = np.mean(raws),  (np.std(raws,  ddof=1) if n > 1 else 0.0)
            norm_m, norm_s = np.mean(norms), (np.std(norms, ddof=1) if n > 1 else 0.0)

            rows.append({
                "dataset_name":              ds,
                "resolved_env_id":           env_id,
                "algorithm":                 algo,
                "n_runs":                    n,
                "mean_raw_return":           f"{raw_m:.2f}",
                "std_raw_return":            f"{raw_s:.2f}",
                "ci95_raw_return_low":       f"{raw_ci[0]:.2f}",
                "ci95_raw_return_high":      f"{raw_ci[1]:.2f}",
                "mean_normalized_score":     f"{norm_m:.2f}",
                "std_normalized_score":      f"{norm_s:.2f}",
                "ci95_normalized_score_low": f"{norm_ci[0]:.2f}",
                "ci95_normalized_score_high":f"{norm_ci[1]:.2f}",
                "notes":                     f"d3rlpy benchmark; {ALGO_N_STEPS} steps; {n} seeds",
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Clean Phase 12 重执行版: Hopper minimum benchmark")
    print(f"  3 datasets x 4 algos x {len(BENCHMARK_SEEDS)} seeds = {TOTAL_RUNS} runs")
    print("=" * 66)
    print()

    # ── Pre-flight A: discrete pipeline complete ──────────────────────────
    print("-- Pre-flight A: discrete pipeline ----------------------------------")
    with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 13, f"audit: {len(audit_rows)} rows, expected 13"
    assert all(r["freeze_ready"] == "yes" for r in audit_rows)
    print(f"  audit: 13 rows, all freeze_ready=yes")

    for label, path, expected in [
        ("main",       MAIN_SUMMARY,    160),
        ("validation", VAL_SUMMARY,     160),
        ("quality",    QUALITY_SUMMARY, 200),
    ]:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == expected, f"{label}: {len(rows)} != {expected}"
        assert all(r["status"] == "completed" for r in rows)
        for r in rows[:5]:
            cp = r.get("checkpoint_path", "").strip()
            assert cp, f"{label}: empty checkpoint_path in row"
            abs_cp = resolve_path(cp)
            assert os.path.isfile(abs_cp) or os.path.isdir(abs_cp), \
                f"{label}: missing checkpoint {cp}"
        print(f"  {label}: {expected} rows completed, 5 ckpts spot-checked OK")

    with open(MECHANISM_SUMMARY, newline="", encoding="utf-8") as f:
        mech_rows = list(csv.DictReader(f))
    assert len(mech_rows) == 18, f"mechanism: {len(mech_rows)} rows, expected 18"
    print(f"  mechanism: 18 rows OK")
    print()

    # ── Pre-flight B: benchmark backend ──────────────────────────────────
    print("-- Pre-flight B: benchmark backend ----------------------------------")
    import d3rlpy
    print(f"  backend: d3rlpy {d3rlpy.__version__}")
    print(f"  log suppression: {'NoopAdapterFactory available' if _NOOP_LOGGER else 'UNAVAILABLE — fallback mode'}")
    for ds_name, info in BENCHMARK_DATASETS.items():
        hdf5 = download_dataset(ds_name)
        print(f"  dataset: {ds_name} → env_id={info['env_id']}")
    print(f"  algorithms: {ALGORITHMS}")
    print(f"  seeds: {BENCHMARK_SEEDS}")
    print(f"  eval_episodes: {BENCHMARK_EVAL_EPISODES}")
    print(f"  n_steps: {ALGO_N_STEPS}")
    print(f"  BC_CFG:    {BC_CFG}")
    print(f"  CQL_CFG:   {CQL_CFG}")
    print(f"  IQL_CFG:   {IQL_CFG}")
    print(f"  TD3BC_CFG: {TD3BC_CFG}")
    print()

    # ── Decide mode ───────────────────────────────────────────────────────
    complete, _ = existing_benchmark_complete()
    os.makedirs(BENCH_DIR, exist_ok=True)
    completed = load_completed_runs()

    n_valid = sum(
        1 for ds in BENCHMARK_DATASETS
        for algo in ALGORITHMS
        for s in BENCHMARK_SEEDS
        if run_is_valid(completed.get((ds, algo, str(s)), {}))
    )

    if complete:
        print(f"-- VERIFY mode: all {TOTAL_RUNS} runs complete — read-only re-verification ---")
    else:
        print(f"-- RESUME/TRAIN mode: {n_valid}/{TOTAL_RUNS} already done, running rest ------")
    print()

    # ── Training / resume loop ────────────────────────────────────────────
    if not complete:
        global_idx = 0
        for ds_name, ds_info in BENCHMARK_DATASETS.items():
            hdf5    = download_dataset(ds_name)
            dataset = load_d4rl_dataset(hdf5)
            env_id  = ds_info["env_id"]
            print(f"-- {ds_name} ({dataset.transition_count} transitions) --")

            for algo_name in ALGORITHMS:
                for seed in BENCHMARK_SEEDS:
                    global_idx += 1
                    key = (ds_name, algo_name, str(seed))

                    if run_is_valid(completed.get(key, {})):
                        print(f"  [{global_idx}/{TOTAL_RUNS}] {algo_name} seed{seed}: SKIP")
                        continue

                    tag = f"hopper_{ds_name.replace('-','_')}_{algo_name}_seed{seed}"
                    print(f"  [{global_idx}/{TOTAL_RUNS}] {tag}...", end="", flush=True)

                    raw_ret    = float("nan")
                    norm_score = float("nan")
                    status     = "failed"
                    notes      = ""
                    ckpt_abs   = ""
                    algo       = None

                    try:
                        algo = create_algo(algo_name, seed)

                        fit_kw = {"n_steps": ALGO_N_STEPS, "show_progress": False}
                        if _NOOP_LOGGER is not None:
                            fit_kw["logger_adapter"] = _NOOP_LOGGER
                        algo.fit(dataset, **fit_kw)

                        ckpt_abs = os.path.join(BENCH_DIR, f"{tag}.d3")
                        algo.save(ckpt_abs)

                        ep_returns = evaluate_policy(algo, env_id, BENCHMARK_EVAL_EPISODES)
                        raw_ret    = float(np.mean(ep_returns))
                        norm_score = normalize_score(raw_ret)
                        status     = "completed"
                        notes      = f"d3rlpy {d3rlpy.__version__}; Hopper min bench"
                        print(f" raw={raw_ret:.1f}  norm={norm_score:.1f}")

                    except Exception as e:
                        notes = f"error: {e}"
                        print(f" FAILED: {e}")

                    finally:
                        if algo is not None:
                            del algo
                        gc.collect()
                        cleanup_backend_scratch()

                    # Durability: append immediately after each run
                    row = {
                        "dataset_name":          ds_name,
                        "resolved_env_id":       env_id,
                        "algorithm":             algo_name,
                        "train_seed":            str(seed),
                        "num_updates_or_epochs": str(ALGO_N_STEPS),
                        "eval_episodes":         str(BENCHMARK_EVAL_EPISODES),
                        "raw_return":            f"{raw_ret:.4f}" if math.isfinite(raw_ret) else str(raw_ret),
                        "normalized_score":      f"{norm_score:.4f}" if math.isfinite(norm_score) else str(norm_score),
                        "checkpoint_path":       normalize_ckpt_path(ckpt_abs) if ckpt_abs else "",
                        "status":                status,
                        "notes":                 notes,
                    }
                    append_seed_row(row)
                    completed[key] = row  # update in-memory dict

            print()

    # ── Finalize seed CSV (deduplicate, canonical order) ──────────────────
    completed = load_completed_runs()
    clean_rows = [
        completed[(ds, algo, str(s))]
        for ds in BENCHMARK_DATASETS
        for algo in ALGORITHMS
        for s in BENCHMARK_SEEDS
        if (ds, algo, str(s)) in completed
    ]
    if len(clean_rows) == TOTAL_RUNS:
        with open(SEED_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SEED_COLUMNS)
            w.writeheader()
            w.writerows(clean_rows)
        print(f"  Seed CSV finalized: {len(clean_rows)} rows")
    print()

    # ── Gate evaluation ───────────────────────────────────────────────────
    print("-- Gate evaluation -------------------------------------------------")
    completed = load_completed_runs()
    gate = {}

    gate["exactly_60_rows"] = len(completed) == TOTAL_RUNS
    gate["all_runs_valid"]  = all(
        run_is_valid(completed.get((ds, algo, str(s)), {}))
        for ds in BENCHMARK_DATASETS
        for algo in ALGORITHMS
        for s in BENCHMARK_SEEDS
    )
    gate["all_norm_finite"] = all(
        math.isfinite(float(r.get("normalized_score", "nan")))
        for r in completed.values()
        if r.get("status") == "completed"
    )
    gate["all_groups_correct"] = all(
        sum(
            1 for s in BENCHMARK_SEEDS
            if run_is_valid(completed.get((ds, algo, str(s)), {}))
        ) == len(BENCHMARK_SEEDS)
        for ds in BENCHMARK_DATASETS
        for algo in ALGORITHMS
    )
    gate["no_scratch_logs"] = (
        not os.path.isdir(SCRATCH_DIR) and
        not os.path.isdir(os.path.join(PROJECT_ROOT, "d3rlpy_logs")) and
        not os.path.isdir(os.path.join(os.getcwd(), "d3rlpy_logs"))
    )

    for k, v in gate.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")

    go = all(gate.values())
    print()

    # ── Aggregate summary ─────────────────────────────────────────────────
    if go:
        summary_rows = rebuild_summary(completed)
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"  Summary CSV: {SUMMARY_CSV} ({len(summary_rows)} rows)")
        print()

        agg = {(r["dataset_name"], r["algorithm"]): float(r["mean_normalized_score"])
               for r in summary_rows}

        print("-- Per-algorithm dataset ordering -----------------------------------")
        for algo in ALGORITHMS:
            scores = sorted(
                [(ds, agg.get((ds, algo), 0.0)) for ds in BENCHMARK_DATASETS],
                key=lambda x: x[1]
            )
            print(f"  {algo}: " + " < ".join(f"{ds}={s:.1f}" for ds, s in scores))

        print()
        print("-- Per-dataset algorithm ranking ------------------------------------")
        for ds in BENCHMARK_DATASETS:
            scores = sorted(
                [(algo, agg.get((ds, algo), 0.0)) for algo in ALGORITHMS],
                key=lambda x: -x[1]
            )
            print(f"  {ds}: " + " > ".join(f"{a}={s:.1f}" for a, s in scores))
        print()

    if go:
        print("Clean Phase 12: PASS")
    else:
        failed = [k for k, v in gate.items() if not v]
        print(f"Clean Phase 12: FAIL — gates failed: {failed}")
        sys.exit(1)
