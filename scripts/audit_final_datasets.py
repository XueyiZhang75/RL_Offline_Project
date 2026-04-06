"""
scripts/audit_final_datasets.py
Clean Phase 6: read-only global audit of 13 active final datasets.

Reads:
  - artifacts/final_datasets/final_dataset_manifest.csv
  - artifacts/behavior_pool/behavior_policy_catalog.csv
  - 13 .npz dataset files

Outputs:
  - artifacts/final_datasets/final_dataset_audit.csv

Does NOT modify any input file.

freeze_ready = yes  iff  Gates A + B + B3 + C + D + E all pass.
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.gridworld_envs import EnvA_v2, EnvB, EnvC

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "artifacts", "final_datasets",
                             "final_dataset_manifest.csv")
CATALOG_PATH  = os.path.join(PROJECT_ROOT, "artifacts", "behavior_pool",
                             "behavior_policy_catalog.csv")
AUDIT_PATH    = os.path.join(PROJECT_ROOT, "artifacts", "final_datasets",
                             "final_dataset_audit.csv")

AUDIT_COLUMNS = [
    "dataset_name", "env_name",
    "file_exists", "load_ok", "integrity_ok",
    "num_transitions", "num_episodes",
    "avg_episode_length", "median_episode_length",
    "avg_return", "std_return", "return_p10", "return_p50", "return_p90",
    "success_rate",
    "unique_state_count", "unique_state_action_count",
    "normalized_state_coverage", "normalized_state_action_coverage",
    "state_visitation_entropy",
    "unique_source_policy_count", "source_policy_entropy",
    "audit_status", "freeze_ready", "notes",
]

EXPECTED_NAMES = {
    "envA_v2_small_wide_medium", "envA_v2_small_narrow_medium",
    "envA_v2_large_wide_medium", "envA_v2_large_narrow_medium",
    "envA_v2_quality_random_wide50k", "envA_v2_quality_suboptimal_wide50k",
    "envA_v2_quality_medium_wide50k", "envA_v2_quality_expert_wide50k",
    "envA_v2_quality_mixed_wide50k",
    "envB_small_wide_medium", "envB_large_narrow_medium",
    "envC_small_wide_medium", "envC_large_narrow_medium",
}

REQUIRED_KEYS = {
    "observations", "actions", "rewards", "next_observations",
    "terminals", "truncations", "episode_ids", "timesteps",
    "source_policy_ids", "source_train_seeds", "source_behavior_epsilons",
}

# ── Environment reachable-set caches ──────────────────────────────────────────

_ENV_CACHE = {}

def _get_env_info(env_name):
    if env_name not in _ENV_CACHE:
        if env_name == "EnvA_v2":
            e = EnvA_v2()
        elif env_name == "EnvB":
            e = EnvB()
        elif env_name == "EnvC":
            e = EnvC()
        else:
            raise ValueError(f"Unknown env: {env_name}")
        rs = e.get_reachable_states()
        sa = e.get_reachable_state_action_pairs()
        _ENV_CACHE[env_name] = {"n_states": len(rs), "n_sa": len(sa)}
    return _ENV_CACHE[env_name]


# ── Entropy helper ────────────────────────────────────────────────────────────

def _entropy(counts_dict):
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts_dict.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# ── Catalog loader ────────────────────────────────────────────────────────────

def load_catalog():
    with open(CATALOG_PATH, newline="", encoding="utf-8") as f:
        return {r["policy_id"]: r for r in csv.DictReader(f)}


# ── Single-dataset audit ─────────────────────────────────────────────────────

def audit_one(manifest_row, catalog):
    """Audit one dataset. Returns dict matching AUDIT_COLUMNS (minus freeze_ready)."""
    name     = manifest_row["dataset_name"]
    env_name = manifest_row["env_name"]
    fpath    = os.path.join(PROJECT_ROOT, manifest_row["file_path"])

    row = {"dataset_name": name, "env_name": env_name}
    notes = []

    # file exists
    exists = os.path.isfile(fpath)
    row["file_exists"] = "yes" if exists else "no"
    if not exists:
        row.update({k: "" for k in AUDIT_COLUMNS if k not in row})
        row["load_ok"] = "no"
        row["integrity_ok"] = "no"
        row["audit_status"] = "fail"
        row["notes"] = "file not found"
        return row

    # load
    try:
        d = np.load(fpath, allow_pickle=True)
    except Exception as e:
        row.update({k: "" for k in AUDIT_COLUMNS if k not in row})
        row["load_ok"] = "no"
        row["integrity_ok"] = "no"
        row["audit_status"] = "fail"
        row["notes"] = f"load error: {e}"
        return row
    row["load_ok"] = "yes"

    # integrity
    keys = set(d.files)
    missing = REQUIRED_KEYS - keys
    if missing:
        notes.append(f"missing keys: {missing}")

    n = len(d["observations"]) if "observations" in keys else 0
    for k in REQUIRED_KEYS:
        if k in keys and len(d[k]) != n:
            notes.append(f"length mismatch: {k}")

    # episode timestep check
    if "timesteps" in keys and "episode_ids" in keys and n > 0:
        ep_ids = d["episode_ids"]
        ts = d["timesteps"]
        prev_ep = ep_ids[0]
        if ts[0] != 0:
            notes.append("first timestep != 0")
        for i in range(1, n):
            if ep_ids[i] != prev_ep:
                if ts[i] != 0:
                    notes.append(f"episode {ep_ids[i]} starts at ts={ts[i]}")
                    break
                prev_ep = ep_ids[i]

    # manifest actual_transitions consistency
    try:
        manifest_actual = int(manifest_row.get("actual_transitions", ""))
        if manifest_actual != n:
            notes.append(f"actual_transitions mismatch: manifest={manifest_actual} data={n}")
    except (ValueError, TypeError):
        notes.append("actual_transitions missing or unparseable in manifest")

    # catalog consistency for non-random sources
    if "source_policy_ids" in keys:
        pids = set(str(x) for x in d["source_policy_ids"])
        for pid in pids:
            if pid == "envA_v2_random":
                continue
            if pid not in catalog:
                notes.append(f"source {pid} not in catalog")

    # source_behavior_epsilons vs catalog behavior_epsilon consistency
    if "source_policy_ids" in keys and "source_behavior_epsilons" in keys:
        s_pids = d["source_policy_ids"]
        s_eps  = d["source_behavior_epsilons"]
        checked_pairs = set()
        for i in range(min(len(s_pids), len(s_eps))):
            pid = str(s_pids[i])
            if pid == "envA_v2_random":
                continue
            if pid in checked_pairs:
                continue
            checked_pairs.add(pid)
            if pid in catalog:
                try:
                    cat_eps = float(catalog[pid]["behavior_epsilon"])
                    dat_eps = float(s_eps[i])
                    if abs(dat_eps - cat_eps) > 1e-3:
                        notes.append(
                            f"epsilon mismatch: {pid} data={dat_eps} catalog={cat_eps}")
                except (ValueError, TypeError):
                    notes.append(f"epsilon unparseable for {pid}")

    integrity_ok = len(notes) == 0
    row["integrity_ok"] = "yes" if integrity_ok else "no"

    # ── Content metrics ───────────────────────────────────────────────────
    obs = d["observations"]
    acts = d["actions"]
    rews = d["rewards"]
    terms = d["terminals"]
    ep_ids_arr = d["episode_ids"]
    pids_arr = d["source_policy_ids"]

    row["num_transitions"] = n

    # episodes
    unique_eps = set(int(x) for x in ep_ids_arr)
    n_eps = len(unique_eps)
    row["num_episodes"] = n_eps

    # per-episode metrics (vectorized)
    ep_ids_np = np.asarray(ep_ids_arr, dtype=np.int32)
    rews_np   = np.asarray(rews, dtype=np.float64)
    terms_np  = np.asarray(terms, dtype=bool)
    sorted_eps = sorted(unique_eps)
    ep_to_idx = {e: i for i, e in enumerate(sorted_eps)}
    rets = np.zeros(n_eps, dtype=np.float64)
    lens_arr = np.zeros(n_eps, dtype=np.int32)
    succs_arr = np.zeros(n_eps, dtype=np.float64)
    for i in range(n):
        idx = ep_to_idx[int(ep_ids_np[i])]
        rets[idx] += rews_np[i]
        lens_arr[idx] += 1
        if terms_np[i]:
            succs_arr[idx] = 1.0
    lens = lens_arr.tolist()
    succs = succs_arr.tolist()

    row["avg_episode_length"]    = f"{np.mean(lens):.2f}"
    row["median_episode_length"] = f"{np.median(lens):.2f}"
    row["avg_return"]            = f"{np.mean(rets):.4f}"
    row["std_return"]            = f"{np.std(rets):.4f}"
    row["return_p10"]            = f"{np.percentile(rets, 10):.4f}"
    row["return_p50"]            = f"{np.percentile(rets, 50):.4f}"
    row["return_p90"]            = f"{np.percentile(rets, 90):.4f}"
    row["success_rate"]          = f"{np.mean(succs):.4f}"

    # ── Coverage (vectorized where possible) ─────────────────────────────
    info = _get_env_info(env_name)
    obs_np = np.asarray(obs, dtype=np.int32)
    acts_np = np.asarray(acts, dtype=np.int32)

    if env_name == "EnvC" and obs_np.shape[1] > 2:
        # Extended state: (row, col, has_key) — encode as single int for fast unique
        state_keys = obs_np[:, 0] * 10000 + obs_np[:, 1] * 10 + obs_np[:, 2]
        sa_keys = state_keys * 10 + acts_np
    else:
        # (row, col) — encode as row*1000 + col
        state_keys = obs_np[:, 0] * 1000 + obs_np[:, 1]
        sa_keys = state_keys * 10 + acts_np

    unique_states_set = np.unique(state_keys)
    unique_sa_set     = np.unique(sa_keys)

    row["unique_state_count"]               = len(unique_states_set)
    row["unique_state_action_count"]        = len(unique_sa_set)
    row["normalized_state_coverage"]        = f"{len(unique_states_set) / info['n_states']:.6f}"
    row["normalized_state_action_coverage"] = f"{len(unique_sa_set) / info['n_sa']:.6f}"

    # state visitation entropy (vectorized)
    _, counts = np.unique(state_keys, return_counts=True)
    total_c = counts.sum()
    probs = counts / total_c
    state_ent = -np.sum(probs * np.log2(probs))
    row["state_visitation_entropy"] = f"{state_ent:.4f}"

    # source policy diversity (vectorized via unique)
    unique_pids, pid_cnts = np.unique(np.asarray(pids_arr, dtype=str), return_counts=True)
    row["unique_source_policy_count"] = len(unique_pids)
    pid_total = pid_cnts.sum()
    pid_probs = pid_cnts / pid_total
    pid_ent = -np.sum(pid_probs * np.log2(pid_probs))
    row["source_policy_entropy"] = f"{pid_ent:.4f}"

    row["audit_status"] = "pass" if integrity_ok else "fail"
    row["notes"] = "; ".join(notes) if notes else "ok"
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Clean Phase 6: Global audit of 13 final datasets")
    print("=" * 66)
    print()

    # ── Pre-flight A: manifest ────────────────────────────────────────────
    print("-- Pre-flight A: manifest ------------------------------------------")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    assert len(manifest_rows) == 13, f"Expected 13 rows, got {len(manifest_rows)}"
    env_counts = {}
    for r in manifest_rows:
        env_counts[r["env_name"]] = env_counts.get(r["env_name"], 0) + 1
    print(f"  rows: {len(manifest_rows)}  env_counts: {env_counts}")
    assert env_counts.get("EnvA_v2", 0) == 9
    assert env_counts.get("EnvB", 0) == 2
    assert env_counts.get("EnvC", 0) == 2
    assert "EnvA" not in env_counts, "Old EnvA rows found!"
    _OLD_REBUILD = "EnvA_main" + "_rebuild"
    assert _OLD_REBUILD not in env_counts, "Old rebuild rows found!"

    actual_names = {r["dataset_name"] for r in manifest_rows}
    assert actual_names == EXPECTED_NAMES, f"Name mismatch: {actual_names ^ EXPECTED_NAMES}"
    print("  manifest pre-flight: OK")
    print()

    # ── Pre-flight B: file existence ──────────────────────────────────────
    print("-- Pre-flight B: file existence ------------------------------------")
    all_exist = True
    for r in manifest_rows:
        fp = os.path.join(PROJECT_ROOT, r["file_path"])
        if not os.path.isfile(fp):
            print(f"  MISSING: {r['file_path']}")
            all_exist = False
    assert all_exist, "Some dataset files are missing"
    print("  all 13 files exist: OK")
    print()

    # ── Pre-flight C: catalog ─────────────────────────────────────────────
    print("-- Pre-flight C: catalog -------------------------------------------")
    catalog = load_catalog()
    bc_pids = [pid for pid, r in catalog.items() if r["env_name"] in ("EnvB", "EnvC")]
    v2_pids = [pid for pid, r in catalog.items() if r["env_name"] == "EnvA_v2"]
    print(f"  EnvB/C catalog entries: {len(bc_pids)}")
    print(f"  EnvA_v2 catalog entries: {len(v2_pids)}")
    assert len(bc_pids) > 0
    assert len(v2_pids) > 0
    print("  catalog pre-flight: OK")
    print()

    # ── Audit each dataset ────────────────────────────────────────────────
    print("-- Auditing 13 datasets --------------------------------------------")
    audit_rows = []
    for mr in manifest_rows:
        ar = audit_one(mr, catalog)
        audit_rows.append(ar)
        status = ar["audit_status"]
        sa_cov = ar.get("normalized_state_action_coverage", "N/A")
        sr = ar.get("success_rate", "N/A")
        print(f"  {ar['dataset_name']}: {status}  SA_cov={sa_cov}  SR={sr}")

    # ── Gate A: integrity ─────────────────────────────────────────────────
    print()
    print("-- Gate A: integrity -----------------------------------------------")
    gate_a = all(r["integrity_ok"] == "yes" for r in audit_rows)
    print(f"  [{'OK' if gate_a else 'FAIL'}] all 13 integrity_ok=yes")

    # ── Gate B: main four coverage ────────────────────────────────────────
    print()
    print("-- Gate B: main four coverage ---------------------------------------")
    def _sa_cov(name):
        for r in audit_rows:
            if r["dataset_name"] == name:
                return float(r["normalized_state_action_coverage"])
        return -1

    sw = _sa_cov("envA_v2_small_wide_medium")
    sn = _sa_cov("envA_v2_small_narrow_medium")
    lw = _sa_cov("envA_v2_large_wide_medium")
    ln = _sa_cov("envA_v2_large_narrow_medium")

    gate_b = {
        "sw > sn":        sw > sn,
        "lw > ln":        lw > ln,
        "sw-sn >= 0.01":  (sw - sn) >= 0.01,
        "lw-ln >= 0.005": (lw - ln) >= 0.005,
    }
    print(f"  sw={sw:.4f}  sn={sn:.4f}  diff={sw-sn:.4f}")
    print(f"  lw={lw:.4f}  ln={ln:.4f}  diff={lw-ln:.4f}")
    for k, v in gate_b.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    gate_b_pass = all(gate_b.values())

    # ── Gate B3: entropy direction ────────────────────────────────────────
    print()
    print("-- Gate B3: entropy direction ---------------------------------------")
    def _ent(name):
        for r in audit_rows:
            if r["dataset_name"] == name:
                return float(r["state_visitation_entropy"])
        return -1

    ent_sw = _ent("envA_v2_small_wide_medium")
    ent_sn = _ent("envA_v2_small_narrow_medium")
    ent_lw = _ent("envA_v2_large_wide_medium")
    ent_ln = _ent("envA_v2_large_narrow_medium")

    gate_b3 = {
        "ent(sw) > ent(sn)": ent_sw > ent_sn,
        "ent(lw) > ent(ln)": ent_lw > ent_ln,
    }
    print(f"  ent_sw={ent_sw:.4f}  ent_sn={ent_sn:.4f}")
    print(f"  ent_lw={ent_lw:.4f}  ent_ln={ent_ln:.4f}")
    for k, v in gate_b3.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    gate_b3_pass = all(gate_b3.values())

    # ── Gate C: quality sweep ─────────────────────────────────────────────
    print()
    print("-- Gate C: quality sweep -------------------------------------------")
    def _metric(name, field):
        for r in audit_rows:
            if r["dataset_name"] == name:
                return float(r[field])
        return -999

    q_sr = {}
    q_ret = {}
    q_std = {}
    for q in ["random", "suboptimal", "medium", "expert", "mixed"]:
        dn = f"envA_v2_quality_{q}_wide50k"
        q_sr[q]  = _metric(dn, "success_rate")
        q_ret[q] = _metric(dn, "avg_return")
        q_std[q] = _metric(dn, "std_return")

    gate_c = {
        "SR monotone":         q_sr["random"] < q_sr["suboptimal"] < q_sr["medium"] < q_sr["expert"],
        "ret monotone":        q_ret["random"] < q_ret["suboptimal"] < q_ret["medium"] < q_ret["expert"],
        "mixed SR in range":   q_sr["suboptimal"] < q_sr["mixed"] < q_sr["expert"],
        "mixed std > med std": q_std["mixed"] > q_std["medium"],
    }
    print(f"  SR:  random={q_sr['random']:.3f}  sub={q_sr['suboptimal']:.3f}  "
          f"med={q_sr['medium']:.3f}  exp={q_sr['expert']:.3f}  mixed={q_sr['mixed']:.3f}")
    print(f"  ret: random={q_ret['random']:.3f}  sub={q_ret['suboptimal']:.3f}  "
          f"med={q_ret['medium']:.3f}  exp={q_ret['expert']:.3f}  mixed={q_ret['mixed']:.3f}")
    print(f"  std: med={q_std['medium']:.4f}  mixed={q_std['mixed']:.4f}")
    for k, v in gate_c.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    gate_c_pass = all(gate_c.values())

    # ── Gate D: EnvB/C retained ───────────────────────────────────────────
    print()
    print("-- Gate D: EnvB/C retained datasets --------------------------------")
    gate_d_pass = True
    for dn in ["envB_small_wide_medium", "envB_large_narrow_medium",
                "envC_small_wide_medium", "envC_large_narrow_medium"]:
        for r in audit_rows:
            if r["dataset_name"] == dn:
                ok = (r["integrity_ok"] == "yes"
                      and float(r["success_rate"]) > 0
                      and float(r["normalized_state_action_coverage"]) > 0)
                status = "OK" if ok else "FAIL"
                if not ok:
                    gate_d_pass = False
                print(f"  [{status}] {dn}  SR={r['success_rate']}  SA_cov={r['normalized_state_action_coverage']}")
                break

    # ── Gate E: manifest consistency ──────────────────────────────────────
    print()
    print("-- Gate E: manifest consistency ------------------------------------")
    audit_names = {r["dataset_name"] for r in audit_rows}
    manifest_names = {r["dataset_name"] for r in manifest_rows}
    gate_e_pass = (audit_names == manifest_names and len(audit_rows) == 13)
    print(f"  [{'OK' if gate_e_pass else 'FAIL'}] 13 datasets match manifest")

    # ── freeze_ready ──────────────────────────────────────────────────────
    freeze = gate_a and gate_b_pass and gate_b3_pass and gate_c_pass and gate_d_pass and gate_e_pass

    # Write audit CSV
    for r in audit_rows:
        r["freeze_ready"] = "yes" if freeze else "no"

    with open(AUDIT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        w.writeheader()
        for r in audit_rows:
            w.writerow({k: r.get(k, "") for k in AUDIT_COLUMNS})

    print()
    print("-- Final result ----------------------------------------------------")
    gates_summary = {
        "Gate A (integrity)":        gate_a,
        "Gate B (main four cov)":    gate_b_pass,
        "Gate B3 (entropy dir)":     gate_b3_pass,
        "Gate C (quality sweep)":    gate_c_pass,
        "Gate D (EnvB/C retained)":  gate_d_pass,
        "Gate E (manifest consist)": gate_e_pass,
    }
    for k, v in gates_summary.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print()
    print(f"  freeze_ready = {'yes' if freeze else 'no'}")
    print()

    if freeze:
        print("Clean Phase 6: PASS")
    else:
        failed = [k for k, v in gates_summary.items() if not v]
        print("Clean Phase 6: FAIL")
        print(f"  Failed: {failed}")
        sys.exit(1)
