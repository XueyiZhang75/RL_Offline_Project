"""
tests/test_envA_v2_iql_main.py
Retrofit Phase R2: minimum sufficient tests for IQL main four on EnvA_v2.
"""

import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from scripts.run_envA_v2_iql_main import (
    IQL_MAIN_SEEDS, IQL_MAIN_DATASETS, TOTAL_RUNS,
    SUMMARY_COLUMNS, MAIN_SUMMARY, IQL_DIR,
    run_is_valid, load_completed_runs, existing_main_complete, append_main_row,
    compute_stats, resolve_iql_path,
    check_main_bc_cql_summary_loadable,
    _IQL_CKPT_REQUIRED_KEYS,
    AUDIT_PATH, MANIFEST_PATH, MAIN_BC_CQL_SUMMARY, R1_IQL_SUMMARY_PATH,
    PROJECT_ROOT,
)
from scripts.run_envA_v2_iql_sanity import (
    IQL_CFG as R1_IQL_CFG,
    train_iql, save_iql_checkpoint, load_iql_checkpoint,
)
from scripts.run_envA_v2_sanity import (
    MLP, encode_obs, load_dataset, evaluate, OBS_DIM, N_ACTIONS, EVAL_EPISODES,
)

# Also import IQL_CFG from main script to verify it's the same as R1's
from scripts.run_envA_v2_iql_main import IQL_CFG as MAIN_IQL_CFG


# ── 1. Frozen constants ──────────────────────────────────────────────────────

class TestFrozenConstants:
    def test_seeds_0_to_19(self):
        assert IQL_MAIN_SEEDS == list(range(20))

    def test_exactly_4_datasets(self):
        assert len(IQL_MAIN_DATASETS) == 4

    def test_small_wide_medium(self):
        assert "envA_v2_small_wide_medium" in IQL_MAIN_DATASETS

    def test_small_narrow_medium(self):
        assert "envA_v2_small_narrow_medium" in IQL_MAIN_DATASETS

    def test_large_wide_medium(self):
        assert "envA_v2_large_wide_medium" in IQL_MAIN_DATASETS

    def test_large_narrow_medium(self):
        assert "envA_v2_large_narrow_medium" in IQL_MAIN_DATASETS

    def test_total_runs_80(self):
        assert TOTAL_RUNS == 80

    def test_algorithm_is_iql(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_iql_main")
        src = inspect.getsource(mod)
        assert '"iql"' in src

    def test_obs_dim_900(self):
        assert OBS_DIM == 900

    def test_eval_episodes_100(self):
        assert EVAL_EPISODES == 100


# ── 2. IQL_CFG inherited from R1 ─────────────────────────────────────────────

class TestIQLCfgInheritance:
    def test_cfg_values_match_r1(self):
        for k in R1_IQL_CFG:
            assert k in MAIN_IQL_CFG
            assert MAIN_IQL_CFG[k] == R1_IQL_CFG[k], \
                f"IQL_CFG['{k}'] differs from R1: {MAIN_IQL_CFG[k]} vs {R1_IQL_CFG[k]}"

    def test_cfg_has_all_required_keys(self):
        for k in ["hidden_dims", "batch_size", "num_updates", "gamma",
                  "expectile", "temperature", "actor_lr", "critic_lr",
                  "value_lr", "weight_decay", "target_tau", "adv_clip"]:
            assert k in MAIN_IQL_CFG


# ── 3. Summary schema ────────────────────────────────────────────────────────

class TestSummarySchema:
    def test_required_columns(self):
        for col in ["dataset_name", "algorithm", "train_seed",
                    "final_actor_loss", "final_q_loss", "final_value_loss",
                    "avg_return", "success_rate", "checkpoint_path", "status"]:
            assert col in SUMMARY_COLUMNS

    def test_80_rows_gate(self):
        assert len(IQL_MAIN_DATASETS) * len(IQL_MAIN_SEEDS) == 80


# ── 4. Pre-flight checks ─────────────────────────────────────────────────────

class TestPreFlight:
    def test_audit_freeze_ready(self):
        with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 13
        assert all(r["freeze_ready"] == "yes" for r in rows)

    def test_4_datasets_in_manifest(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_MAIN_DATASETS:
            assert ds in mrows
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            assert os.path.isfile(p)

    def test_dataset_schema_11_keys(self):
        REQUIRED = ["observations", "actions", "rewards", "next_observations",
                    "terminals", "truncations", "episode_ids", "timesteps",
                    "source_policy_ids", "source_train_seeds", "source_behavior_epsilons"]
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_MAIN_DATASETS:
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            d = np.load(p, allow_pickle=True)
            for k in REQUIRED:
                assert k in d, f"{ds} missing key '{k}'"

    def test_dataset_obs_shape_N2(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_MAIN_DATASETS:
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            d = np.load(p, allow_pickle=True)
            assert d["observations"].shape[1] == 2

    def test_bc_cql_main_160_completed(self):
        with open(MAIN_BC_CQL_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        assert all(r["status"] == "completed" for r in rows)

    def test_bc_cql_main_ckpts_loadable_spot(self):
        with open(MAIN_BC_CQL_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Spot-check first 3
        for r in rows[:3]:
            ap = os.path.normpath(r["checkpoint_path"])
            assert os.path.isfile(ap)
            torch.load(ap, map_location="cpu", weights_only=False)

    def test_r1_iql_sanity_6_completed(self):
        with open(R1_IQL_SUMMARY_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        assert all(r["status"] == "completed" for r in rows)

    def test_r1_iql_sanity_ckpts_loadable(self):
        with open(R1_IQL_SUMMARY_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            ap = resolve_iql_path(r["checkpoint_path"])
            assert os.path.isfile(ap)
            torch.load(ap, map_location="cpu", weights_only=False)


# ── 5. Resume / reuse mode logic ─────────────────────────────────────────────

class TestResumeModeLogic:
    def _make_row(self, ds, seed, status="completed", cp="fake/path.pt", ret="0.5", sr="1.0"):
        r = {c: "" for c in SUMMARY_COLUMNS}
        r.update({"dataset_name": ds, "algorithm": "iql", "train_seed": str(seed),
                  "status": status, "checkpoint_path": cp,
                  "avg_return": ret, "success_rate": sr})
        return r

    def test_run_is_valid_false_no_file(self):
        row = self._make_row("envA_v2_small_wide_medium", 0, cp="nonexistent/ghost.pt")
        assert not run_is_valid(row)

    def test_run_is_valid_false_failed_status(self):
        row = self._make_row("envA_v2_small_wide_medium", 0, status="failed")
        assert not run_is_valid(row)

    def test_run_is_valid_false_nan_return(self):
        row = self._make_row("envA_v2_small_wide_medium", 0, ret="nan")
        assert not run_is_valid(row)

    def test_existing_main_complete_false_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.run_envA_v2_iql_main.MAIN_SUMMARY",
                            str(tmp_path / "nonexistent.csv"))
        ok, _ = existing_main_complete()
        assert not ok

    def test_existing_main_complete_false_partial(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "partial.csv")
        monkeypatch.setattr("scripts.run_envA_v2_iql_main.MAIN_SUMMARY", csv_path)
        for i in range(3):
            row = {c: "" for c in SUMMARY_COLUMNS}
            row.update({"status": "completed", "checkpoint_path": "fake.pt",
                        "algorithm": "iql", "dataset_name": IQL_MAIN_DATASETS[0],
                        "train_seed": str(i), "avg_return": "0.5", "success_rate": "1.0"})
            append_main_row(row)
        ok, _ = existing_main_complete()
        assert not ok


# ── 6. Append-after-each-run ─────────────────────────────────────────────────

class TestAppendAfterRun:
    def test_creates_with_header(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "main.csv")
        monkeypatch.setattr("scripts.run_envA_v2_iql_main.MAIN_SUMMARY", csv_path)
        row = {c: f"v{i}" for i, c in enumerate(SUMMARY_COLUMNS)}
        append_main_row(row)
        assert os.path.isfile(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 row

    def test_appends_multiple_rows(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "main.csv")
        monkeypatch.setattr("scripts.run_envA_v2_iql_main.MAIN_SUMMARY", csv_path)
        for i in range(5):
            row = {c: str(i) for c in SUMMARY_COLUMNS}
            append_main_row(row)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5


# ── 7. Checkpoint roundtrip ──────────────────────────────────────────────────

class TestCheckpointRoundtrip:
    def _make_tiny_data(self, n=200):
        obs   = encode_obs(np.random.randint(0, 30, (n, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (n,)).astype(np.int64)
        rews  = np.random.randn(n).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (n, 2)).astype(np.int32))
        terms = (np.random.rand(n) < 0.05).astype(np.float32)
        return obs, acts, rews, nobs, terms

    def test_save_load_roundtrip(self, tmp_path):
        obs, acts, rews, nobs, terms = self._make_tiny_data()
        tiny_cfg = {**R1_IQL_CFG, "num_updates": 5, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, tiny_cfg, 0)
        path = str(tmp_path / "test.pt")
        save_iql_checkpoint(path, actor, q1, q2, v, "test_ds", 0, tiny_cfg, al, ql, vl)
        loaded_actor, ckpt = load_iql_checkpoint(path)
        for k in ["actor_state_dict", "q1_state_dict", "q2_state_dict", "value_state_dict"]:
            assert k in ckpt


# ── 8. Smoke test ────────────────────────────────────────────────────────────

class TestSmoke:
    def test_iql_smoke_finite_losses(self):
        obs   = encode_obs(np.random.randint(0, 30, (300, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (300,)).astype(np.int64)
        rews  = np.random.randn(300).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (300, 2)).astype(np.int32))
        terms = (np.random.rand(300) < 0.05).astype(np.float32)
        tiny_cfg = {**R1_IQL_CFG, "num_updates": 10, "batch_size": 32}
        _, _, _, _, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, tiny_cfg, 0)
        assert math.isfinite(al) and math.isfinite(ql) and math.isfinite(vl)


# ── 9. Aggregate stats helper ────────────────────────────────────────────────

class TestComputeStats:
    def test_returns_mean_std_ci(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        m, s, ci = compute_stats(values)
        assert math.isfinite(m)
        assert math.isfinite(s)
        assert math.isfinite(ci[0]) and math.isfinite(ci[1])
        assert ci[0] <= m <= ci[1]

    def test_single_value_no_crash(self):
        m, s, ci = compute_stats([0.42])
        assert math.isfinite(m)
        assert ci == (pytest.approx(0.42), pytest.approx(0.42))

    def test_zero_std_no_crash(self):
        m, s, ci = compute_stats([0.5, 0.5, 0.5])
        assert m == pytest.approx(0.5)


# ── 10. No old-chain imports ─────────────────────────────────────────────────

class TestNoOldChain:
    def test_no_old_imports(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_iql_main")
        src = inspect.getsource(mod)
        assert "EnvA_main_rebuild" not in src
        assert "run_hopper_benchmark" not in src
        assert "train_behavior_pool" not in src
        assert "envB" not in src.lower().split("envbc")[0] if "envbc" in src.lower() else True

    def test_no_manifest_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_iql_main")
        src = inspect.getsource(mod)
        assert 'open(MANIFEST_PATH, "w"' not in src

    def test_no_audit_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_iql_main")
        src = inspect.getsource(mod)
        assert 'open(AUDIT_PATH, "w"' not in src


# ── 11. BC/CQL main summary loadability helper (patch) ───────────────────────

class TestBCCQLMainLoadabilityHelper:
    def test_real_bc_cql_main_ckpts_loadable_spot(self):
        # Spot-check first 3 via the helper logic (not full 160 to save test time)
        with open(MAIN_BC_CQL_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        for r in rows[:3]:
            ap = os.path.normpath(r["checkpoint_path"])
            assert os.path.isfile(ap)
            torch.load(ap, map_location="cpu", weights_only=False)

    def test_helper_detects_missing_file(self, tmp_path):
        fake_csv = str(tmp_path / "fake_main.csv")
        cols = ["dataset_name", "algorithm", "train_seed", "num_updates",
                "final_train_loss", "eval_episodes", "avg_return", "success_rate",
                "avg_episode_length", "checkpoint_path", "status", "notes"]
        rows = []
        for i in range(160):
            rows.append({c: "x" for c in cols})
            rows[-1].update({"status": "completed",
                             "checkpoint_path": str(tmp_path / f"ghost_{i}.pt")})
        with open(fake_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(AssertionError):
            check_main_bc_cql_summary_loadable(fake_csv)

    def test_helper_detects_empty_checkpoint_path(self, tmp_path):
        fake_csv = str(tmp_path / "fake_main_empty.csv")
        cols = ["dataset_name", "algorithm", "train_seed", "num_updates",
                "final_train_loss", "eval_episodes", "avg_return", "success_rate",
                "avg_episode_length", "checkpoint_path", "status", "notes"]
        rows = []
        for i in range(160):
            rows.append({c: "x" for c in cols})
            rows[-1].update({"status": "completed", "checkpoint_path": ""})
        with open(fake_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(AssertionError):
            check_main_bc_cql_summary_loadable(fake_csv)

    def test_helper_detects_corrupt_checkpoint(self, tmp_path):
        fake_csv = str(tmp_path / "fake_main_corrupt.csv")
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"not a real checkpoint")
        cols = ["dataset_name", "algorithm", "train_seed", "num_updates",
                "final_train_loss", "eval_episodes", "avg_return", "success_rate",
                "avg_episode_length", "checkpoint_path", "status", "notes"]
        rows = []
        for i in range(160):
            rows.append({c: "x" for c in cols})
            rows[-1].update({"status": "completed", "checkpoint_path": corrupt})
        with open(fake_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises((AssertionError, Exception)):
            check_main_bc_cql_summary_loadable(fake_csv)


# ── 12. run_is_valid strict definition (patch) ───────────────────────────────

class TestRunIsValidStrict:
    def _make_valid_ckpt(self, tmp_path):
        """Create a real minimal IQL checkpoint with all required keys."""
        obs   = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**R1_IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, cfg, 0)
        path  = str(tmp_path / "iql_valid.pt")
        save_iql_checkpoint(path, actor, q1, q2, v, "test_ds", 0, cfg, al, ql, vl)
        return path

    def _make_row(self, cp, status="completed", ret="0.41", sr="1.0"):
        r = {c: "" for c in SUMMARY_COLUMNS}
        r.update({"status": status, "checkpoint_path": cp,
                  "avg_return": ret, "success_rate": sr, "algorithm": "iql"})
        return r

    def test_valid_ckpt_and_finite_metrics_is_true(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        row  = self._make_row(path)
        assert run_is_valid(row)

    def test_missing_file_is_false(self):
        row = self._make_row("nonexistent/ghost.pt")
        assert not run_is_valid(row)

    def test_corrupt_ckpt_is_false(self, tmp_path):
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"garbage bytes")
        row = self._make_row(corrupt)
        assert not run_is_valid(row)

    def test_missing_actor_state_dict_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        del ckpt["actor_state_dict"]
        torch.save(ckpt, path)
        assert not run_is_valid(self._make_row(path))

    def test_missing_q1_state_dict_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        del ckpt["q1_state_dict"]
        torch.save(ckpt, path)
        assert not run_is_valid(self._make_row(path))

    def test_missing_value_state_dict_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        del ckpt["value_state_dict"]
        torch.save(ckpt, path)
        assert not run_is_valid(self._make_row(path))

    def test_nan_avg_return_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        assert not run_is_valid(self._make_row(path, ret="nan"))

    def test_nan_success_rate_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        assert not run_is_valid(self._make_row(path, sr="nan"))

    def test_failed_status_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        assert not run_is_valid(self._make_row(path, status="failed"))


# ── 13. existing_main_complete stricter validity (patch) ─────────────────────

class TestExistingMainCompleteStrict:
    def _fill_csv(self, csv_path, ckpt_path_fn, status="completed", ret="0.41", sr="1.0"):
        """Write a 80-row fake summary using ckpt_path_fn(ds, seed) for paths."""
        rows = []
        for ds in IQL_MAIN_DATASETS:
            for s in IQL_MAIN_SEEDS:
                r = {c: "" for c in SUMMARY_COLUMNS}
                r.update({"dataset_name": ds, "algorithm": "iql",
                          "train_seed": str(s), "status": status,
                          "checkpoint_path": ckpt_path_fn(ds, s),
                          "avg_return": ret, "success_rate": sr})
                rows.append(r)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(rows)

    def test_complete_with_valid_ckpts_is_true(self, tmp_path, monkeypatch):
        # Create one real IQL ckpt and reuse for all 80 rows
        obs   = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**R1_IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, cfg, 0)
        ckpt  = str(tmp_path / "shared.pt")
        save_iql_checkpoint(ckpt, actor, q1, q2, v, "ds", 0, cfg, al, ql, vl)

        csv_path = str(tmp_path / "main.csv")
        self._fill_csv(csv_path, lambda ds, s: ckpt)
        monkeypatch.setattr("scripts.run_envA_v2_iql_main.MAIN_SUMMARY", csv_path)
        ok, _ = existing_main_complete()
        assert ok

    def test_corrupt_ckpt_makes_incomplete(self, tmp_path, monkeypatch):
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"garbage")
        csv_path = str(tmp_path / "main.csv")
        self._fill_csv(csv_path, lambda ds, s: corrupt)
        monkeypatch.setattr("scripts.run_envA_v2_iql_main.MAIN_SUMMARY", csv_path)
        ok, _ = existing_main_complete()
        assert not ok

    def test_missing_key_in_ckpt_makes_incomplete(self, tmp_path, monkeypatch):
        obs   = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**R1_IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, cfg, 0)
        ckpt_path = str(tmp_path / "stripped.pt")
        save_iql_checkpoint(ckpt_path, actor, q1, q2, v, "ds", 0, cfg, al, ql, vl)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        del ckpt["q2_state_dict"]
        torch.save(ckpt, ckpt_path)

        csv_path = str(tmp_path / "main.csv")
        self._fill_csv(csv_path, lambda ds, s: ckpt_path)
        monkeypatch.setattr("scripts.run_envA_v2_iql_main.MAIN_SUMMARY", csv_path)
        ok, _ = existing_main_complete()
        assert not ok
