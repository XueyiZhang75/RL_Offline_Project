"""
tests/test_envbc_iql_validation.py
Retrofit Phase R3: minimum sufficient tests for IQL EnvB/C validation.
"""

import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from scripts.run_envbc_iql_validation import (
    VALIDATION_SEEDS, IQL_VALIDATION_DATASETS, TOTAL_RUNS,
    SUMMARY_COLUMNS, SUMMARY_PATH, IQL_DIR,
    DS_ENV_MAP, REQUIRED_NPZ_KEYS, _IQL_CKPT_REQUIRED_KEYS,
    run_is_valid, load_completed_runs, existing_validation_complete, append_val_row,
    resolve_val_path, compute_stats,
    check_bc_cql_val_summary_loadable, check_iql_main_summary_valid,
    train_iql_envbc, save_iql_checkpoint_envbc,
    capture_frozen_file_snapshots, frozen_snapshots_equal, FROZEN_FILES,
    AUDIT_PATH, MANIFEST_PATH, BC_CQL_VAL_SUM, IQL_MAIN_SUM, PROJECT_ROOT,
)
from scripts.run_envA_v2_iql_sanity import (
    IQL_CFG, train_iql, save_iql_checkpoint,
)
from scripts.run_envbc_validation import (
    ENVB_OBS_DIM, ENVC_OBS_DIM, encode_envB, encode_envC,
)


# ── A. Frozen constants ──────────────────────────────────────────────────────

class TestFrozenConstants:
    def test_validation_seeds_0_to_19(self):
        assert VALIDATION_SEEDS == list(range(20))

    def test_exactly_4_datasets(self):
        assert len(IQL_VALIDATION_DATASETS) == 4

    def test_envB_small_wide(self):
        assert "envB_small_wide_medium" in IQL_VALIDATION_DATASETS

    def test_envB_large_narrow(self):
        assert "envB_large_narrow_medium" in IQL_VALIDATION_DATASETS

    def test_envC_small_wide(self):
        assert "envC_small_wide_medium" in IQL_VALIDATION_DATASETS

    def test_envC_large_narrow(self):
        assert "envC_large_narrow_medium" in IQL_VALIDATION_DATASETS

    def test_total_runs_80(self):
        assert TOTAL_RUNS == 80

    def test_algorithm_is_iql(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envbc_iql_validation")
        src = inspect.getsource(mod)
        assert '"iql"' in src

    def test_envB_obs_dim_225(self):
        assert ENVB_OBS_DIM == 225

    def test_envC_obs_dim_450(self):
        assert ENVC_OBS_DIM == 450


# ── B. Pre-flight / schema ───────────────────────────────────────────────────

class TestPreFlightSchema:
    def test_retained_datasets_in_manifest(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_VALIDATION_DATASETS:
            assert ds in mrows
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            assert os.path.isfile(p)

    def test_11key_schema_real_datasets(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_VALIDATION_DATASETS:
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            d = np.load(p, allow_pickle=True)
            for k in REQUIRED_NPZ_KEYS:
                assert k in d, f"{ds} missing key '{k}'"

    def test_real_datasets_length_consistent(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_VALIDATION_DATASETS:
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            d = np.load(p, allow_pickle=True)
            n = len(d["observations"])
            for k in REQUIRED_NPZ_KEYS[1:]:
                assert len(d[k]) == n, f"{ds}: {k} length mismatch"

    def test_envB_obs_shape_N2(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in ["envB_small_wide_medium", "envB_large_narrow_medium"]:
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            d = np.load(p, allow_pickle=True)
            assert d["observations"].shape[1] == 2, \
                f"{ds} obs shape {d['observations'].shape}, expected (N,2)"

    def test_envC_obs_shape_N3(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in ["envC_small_wide_medium", "envC_large_narrow_medium"]:
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            d = np.load(p, allow_pickle=True)
            assert d["observations"].shape[1] == 3, \
                f"{ds} obs shape {d['observations'].shape}, expected (N,3)"

    def test_wrong_envB_obs_shape_detectable(self, tmp_path):
        fake = str(tmp_path / "fake.npz")
        np.savez(fake, observations=np.zeros((10, 3), dtype=np.int32))
        d = np.load(fake)
        assert d["observations"].shape[1] != 2

    def test_wrong_envC_obs_shape_detectable(self, tmp_path):
        fake = str(tmp_path / "fake.npz")
        np.savez(fake, observations=np.zeros((10, 2), dtype=np.int32))
        d = np.load(fake)
        assert d["observations"].shape[1] != 3

    def test_length_mismatch_detectable(self, tmp_path):
        fake = str(tmp_path / "fake.npz")
        np.savez(fake, observations=np.zeros((10, 2), dtype=np.int32),
                 actions=np.zeros(5))  # wrong length
        d = np.load(fake)
        assert len(d["observations"]) != len(d["actions"])


# ── C. BC/CQL validation prerequisite helper ─────────────────────────────────

class TestBCCQLValHelper:
    def test_real_bc_cql_val_spot_check(self):
        with open(BC_CQL_VAL_SUM, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        for r in rows[:3]:
            ap = resolve_val_path(r["checkpoint_path"])
            assert os.path.isfile(ap)
            torch.load(ap, map_location="cpu", weights_only=False)

    def _fake_val_csv(self, tmp_path, cp_fn, n=160):
        fake = str(tmp_path / "fake_val.csv")
        cols = ["env_name", "dataset_name", "algorithm", "train_seed", "obs_dim",
                "num_updates", "final_train_loss", "eval_episodes", "avg_return",
                "success_rate", "avg_episode_length", "checkpoint_path", "status", "notes"]
        rows = [{c: "x" for c in cols} for _ in range(n)]
        for i, r in enumerate(rows):
            r["status"] = "completed"
            r["checkpoint_path"] = cp_fn(i)
        with open(fake, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return fake

    def test_helper_detects_missing_file(self, tmp_path):
        fake = self._fake_val_csv(tmp_path, lambda i: str(tmp_path / f"ghost_{i}.pt"))
        with pytest.raises(AssertionError):
            check_bc_cql_val_summary_loadable(fake)

    def test_helper_detects_empty_path(self, tmp_path):
        fake = self._fake_val_csv(tmp_path, lambda i: "")
        with pytest.raises(AssertionError):
            check_bc_cql_val_summary_loadable(fake)

    def test_helper_detects_corrupt_checkpoint(self, tmp_path):
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"not a checkpoint")
        fake = self._fake_val_csv(tmp_path, lambda i: corrupt)
        with pytest.raises((AssertionError, Exception)):
            check_bc_cql_val_summary_loadable(fake)


# ── D. R2 IQL main prerequisite helper ───────────────────────────────────────

class TestIQLMainPrerequisiteHelper:
    def _make_iql_ckpt(self, tmp_path, strip_key=None):
        # Use 225-dim EnvB-like data with train_iql_envbc
        obs   = encode_envB(np.random.randint(0, 15, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_envB(np.random.randint(0, 15, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql_envbc(
            obs, acts, rews, nobs, terms, cfg, 0, ENVB_OBS_DIM)
        path  = str(tmp_path / "test_main.pt")
        save_iql_checkpoint_envbc(path, actor, q1, q2, v, "ds", 0, cfg, al, ql, vl, ENVB_OBS_DIM)
        if strip_key:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            del ckpt[strip_key]
            torch.save(ckpt, path)
        return path

    def _fake_main_csv(self, tmp_path, cp_fn, n=80):
        fake = str(tmp_path / "fake_main.csv")
        from scripts.run_envA_v2_iql_main import SUMMARY_COLUMNS as MC
        rows = [{c: "x" for c in MC} for _ in range(n)]
        for r in rows:
            r["status"] = "completed"
            r["avg_return"] = "0.41"
            r["success_rate"] = "1.0"
            r["checkpoint_path"] = cp_fn()
        with open(fake, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MC)
            w.writeheader()
            w.writerows(rows)
        return fake

    def test_real_iql_main_spot_loadable(self):
        with open(IQL_MAIN_SUM, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 80
        for r in rows[:3]:
            ap = resolve_val_path(r["checkpoint_path"])
            assert os.path.isfile(ap)
            ckpt = torch.load(ap, map_location="cpu", weights_only=False)
            for k in _IQL_CKPT_REQUIRED_KEYS:
                assert k in ckpt

    def test_helper_detects_missing_ckpt(self, tmp_path):
        fake = self._fake_main_csv(tmp_path, lambda: str(tmp_path / "ghost.pt"))
        with pytest.raises(AssertionError):
            check_iql_main_summary_valid(fake)

    def test_helper_detects_corrupt_ckpt(self, tmp_path):
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"garbage")
        fake = self._fake_main_csv(tmp_path, lambda: corrupt)
        with pytest.raises((AssertionError, Exception)):
            check_iql_main_summary_valid(fake)

    def test_helper_detects_missing_actor_key(self, tmp_path):
        path = self._make_iql_ckpt(tmp_path, strip_key="actor_state_dict")
        fake = self._fake_main_csv(tmp_path, lambda: path)
        with pytest.raises(AssertionError):
            check_iql_main_summary_valid(fake)

    def test_helper_detects_missing_q1_key(self, tmp_path):
        path = self._make_iql_ckpt(tmp_path, strip_key="q1_state_dict")
        fake = self._fake_main_csv(tmp_path, lambda: path)
        with pytest.raises(AssertionError):
            check_iql_main_summary_valid(fake)


# ── E. Resume / verify logic ─────────────────────────────────────────────────

class TestResumeModeLogic:
    def _make_valid_ckpt(self, tmp_path):
        obs   = encode_envB(np.random.randint(0, 15, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_envB(np.random.randint(0, 15, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql_envbc(
            obs, acts, rews, nobs, terms, cfg, 0, ENVB_OBS_DIM)
        path  = str(tmp_path / "valid_ckpt.pt")
        save_iql_checkpoint_envbc(path, actor, q1, q2, v, "ds", 0, cfg, al, ql, vl, ENVB_OBS_DIM)
        return path

    def _row(self, cp, status="completed", ret="0.76", sr="1.0"):
        r = {c: "" for c in SUMMARY_COLUMNS}
        r.update({"status": status, "checkpoint_path": cp,
                  "avg_return": ret, "success_rate": sr, "algorithm": "iql"})
        return r

    def test_valid_run_is_valid_true(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        assert run_is_valid(self._row(path))

    def test_missing_file_is_false(self):
        assert not run_is_valid(self._row("nonexistent/ghost.pt"))

    def test_corrupt_ckpt_is_false(self, tmp_path):
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"garbage")
        assert not run_is_valid(self._row(corrupt))

    def test_nan_avg_return_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        assert not run_is_valid(self._row(path, ret="nan"))

    def test_nan_success_rate_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        assert not run_is_valid(self._row(path, sr="nan"))

    def test_failed_status_is_false(self, tmp_path):
        path = self._make_valid_ckpt(tmp_path)
        assert not run_is_valid(self._row(path, status="failed"))

    def test_existing_validation_complete_false_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.run_envbc_iql_validation.SUMMARY_PATH",
                            str(tmp_path / "nonexistent.csv"))
        ok, _ = existing_validation_complete()
        assert not ok

    def test_existing_validation_complete_false_partial(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "partial.csv")
        monkeypatch.setattr("scripts.run_envbc_iql_validation.SUMMARY_PATH", csv_path)
        for i in range(3):
            row = {c: "" for c in SUMMARY_COLUMNS}
            row.update({"status": "completed", "checkpoint_path": "fake.pt",
                        "algorithm": "iql", "dataset_name": IQL_VALIDATION_DATASETS[0],
                        "train_seed": str(i), "avg_return": "0.5", "success_rate": "1.0"})
            append_val_row(row)
        ok, _ = existing_validation_complete()
        assert not ok

    def test_existing_validation_complete_false_corrupt_ckpt(self, tmp_path, monkeypatch):
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"garbage")
        csv_path = str(tmp_path / "main.csv")
        rows = []
        for ds in IQL_VALIDATION_DATASETS:
            for s in VALIDATION_SEEDS:
                r = {c: "" for c in SUMMARY_COLUMNS}
                r.update({"dataset_name": ds, "algorithm": "iql",
                          "train_seed": str(s), "status": "completed",
                          "checkpoint_path": corrupt,
                          "avg_return": "0.5", "success_rate": "1.0"})
                rows.append(r)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        monkeypatch.setattr("scripts.run_envbc_iql_validation.SUMMARY_PATH", csv_path)
        ok, _ = existing_validation_complete()
        assert not ok

    def test_existing_validation_complete_true_with_valid_ckpts(self, tmp_path, monkeypatch):
        obs   = encode_envB(np.random.randint(0, 15, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_envB(np.random.randint(0, 15, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql_envbc(
            obs, acts, rews, nobs, terms, cfg, 0, ENVB_OBS_DIM)
        shared = str(tmp_path / "shared.pt")
        save_iql_checkpoint_envbc(shared, actor, q1, q2, v, "ds", 0, cfg, al, ql, vl, ENVB_OBS_DIM)

        csv_path = str(tmp_path / "complete.csv")
        rows = []
        for ds in IQL_VALIDATION_DATASETS:
            for s in VALIDATION_SEEDS:
                r = {c: "" for c in SUMMARY_COLUMNS}
                r.update({"dataset_name": ds, "algorithm": "iql",
                          "train_seed": str(s), "status": "completed",
                          "checkpoint_path": shared,
                          "avg_return": "0.5", "success_rate": "1.0"})
                rows.append(r)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        monkeypatch.setattr("scripts.run_envbc_iql_validation.SUMMARY_PATH", csv_path)
        ok, _ = existing_validation_complete()
        assert ok


# ── F. Checkpoint roundtrip ──────────────────────────────────────────────────

class TestCheckpointRoundtrip:
    def test_save_load_roundtrip_envB_like(self, tmp_path):
        obs   = encode_envB(np.random.randint(0, 15, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_envB(np.random.randint(0, 15, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql_envbc(
            obs, acts, rews, nobs, terms, cfg, 0, ENVB_OBS_DIM)
        path = str(tmp_path / "envB_iql.pt")
        save_iql_checkpoint_envbc(path, actor, q1, q2, v, "envB_test", 0, cfg, al, ql, vl, ENVB_OBS_DIM)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for k in _IQL_CKPT_REQUIRED_KEYS:
            assert k in ckpt
        assert ckpt["obs_dim"] == ENVB_OBS_DIM

    def test_save_load_roundtrip_envC_like(self, tmp_path):
        # EnvC: (row, col, has_key) — has_key must be 0 or 1
        raw   = np.column_stack([np.random.randint(0, 15, (200, 2)),
                                 np.random.randint(0, 2, 200)]).astype(np.int32)
        obs   = encode_envC(raw)
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_envC(raw)
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql_envbc(
            obs, acts, rews, nobs, terms, cfg, 0, ENVC_OBS_DIM)
        path = str(tmp_path / "envC_iql.pt")
        save_iql_checkpoint_envbc(path, actor, q1, q2, v, "envC_test", 0, cfg, al, ql, vl, ENVC_OBS_DIM)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for k in _IQL_CKPT_REQUIRED_KEYS:
            assert k in ckpt
        assert ckpt["obs_dim"] == ENVC_OBS_DIM


# ── G. Smoke tests ───────────────────────────────────────────────────────────

class TestSmoke:
    def test_iql_smoke_envB_like(self):
        obs   = encode_envB(np.random.randint(0, 15, (300, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (300,)).astype(np.int64)
        rews  = np.random.randn(300).astype(np.float32)
        nobs  = encode_envB(np.random.randint(0, 15, (300, 2)).astype(np.int32))
        terms = (np.random.rand(300) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 10, "batch_size": 32}
        _, _, _, _, al, ql, vl = train_iql_envbc(
            obs, acts, rews, nobs, terms, cfg, 0, ENVB_OBS_DIM)
        assert math.isfinite(al) and math.isfinite(ql) and math.isfinite(vl)

    def test_iql_smoke_envC_like(self):
        raw   = np.column_stack([np.random.randint(0, 15, (300, 2)),
                                 np.random.randint(0, 2, 300)]).astype(np.int32)
        obs   = encode_envC(raw)
        acts  = np.random.randint(0, 4, (300,)).astype(np.int64)
        rews  = np.random.randn(300).astype(np.float32)
        nobs  = encode_envC(raw)
        terms = (np.random.rand(300) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 10, "batch_size": 32}
        _, _, _, _, al, ql, vl = train_iql_envbc(
            obs, acts, rews, nobs, terms, cfg, 0, ENVC_OBS_DIM)
        assert math.isfinite(al) and math.isfinite(ql) and math.isfinite(vl)

    def test_iql_cfg_inherited_from_r1(self):
        from scripts.run_envA_v2_iql_sanity import IQL_CFG as R1_CFG
        from scripts.run_envbc_iql_validation import IQL_CFG as R3_CFG
        assert R3_CFG is R1_CFG  # same object (direct import)


# ── H. No scope creep ────────────────────────────────────────────────────────

class TestNoScopeCreep:
    def test_no_manifest_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envbc_iql_validation")
        src = inspect.getsource(mod)
        assert 'open(MANIFEST_PATH, "w"' not in src

    def test_no_audit_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envbc_iql_validation")
        src = inspect.getsource(mod)
        assert 'open(AUDIT_PATH, "w"' not in src

    def test_no_old_chain(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envbc_iql_validation")
        src = inspect.getsource(mod)
        assert "EnvA_main_rebuild" not in src
        assert "run_hopper_benchmark" not in src

    def test_no_behavior_catalog_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envbc_iql_validation")
        src = inspect.getsource(mod)
        assert "behavior_policy_catalog" not in src


# ── I. Frozen-file snapshot helpers (patch) ──────────────────────────────────

class TestFrozenFileSnapshot:
    def test_unchanged_files_compare_equal(self, tmp_path):
        f1 = tmp_path / "frozen1.csv"
        f2 = tmp_path / "frozen2.csv"
        f1.write_text("header\nrow1\nrow2\n")
        f2.write_text("header\nrowA\nrowB\n")
        paths = [str(f1), str(f2)]
        before = capture_frozen_file_snapshots(paths)
        after  = capture_frozen_file_snapshots(paths)
        ok, diffs = frozen_snapshots_equal(before, after)
        assert ok, f"Expected no diffs, got: {diffs}"

    def test_modified_content_detected(self, tmp_path):
        f = tmp_path / "frozen.csv"
        f.write_text("original content\n")
        paths = [str(f)]
        before = capture_frozen_file_snapshots(paths)
        f.write_text("MODIFIED content\n")  # change content
        after = capture_frozen_file_snapshots(paths)
        ok, diffs = frozen_snapshots_equal(before, after)
        assert not ok, "Expected diff to be detected"
        assert len(diffs) > 0

    def test_missing_file_detected(self, tmp_path):
        f = tmp_path / "frozen.csv"
        f.write_text("some content\n")
        paths = [str(f)]
        before = capture_frozen_file_snapshots(paths)
        f.unlink()  # delete file
        after = capture_frozen_file_snapshots(paths)
        ok, diffs = frozen_snapshots_equal(before, after)
        assert not ok, "Deleted file must be detected as modified"
        assert len(diffs) > 0

    def test_gate_not_hardcoded_true(self):
        """Verify no_frozen_files_modified gate uses real helper, not literal True."""
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envbc_iql_validation")
        src = inspect.getsource(mod)
        # The gate must NOT be a hardcoded True assignment
        assert 'gate["no_frozen_files_modified"] = True' not in src
        # The gate must reference the snapshot comparison
        assert "frozen_snapshots_equal" in src
