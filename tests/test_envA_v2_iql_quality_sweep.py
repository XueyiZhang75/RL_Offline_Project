"""
tests/test_envA_v2_iql_quality_sweep.py
Retrofit Phase R4: minimum sufficient tests for IQL quality sweep on EnvA_v2.
"""

import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from scripts.run_envA_v2_iql_quality_sweep import (
    IQL_QUALITY_SEEDS, IQL_QUALITY_DATASETS, TOTAL_RUNS,
    SUMMARY_COLUMNS, SUMMARY_PATH, IQL_DIR,
    QUALITY_BIN_MAP, REQUIRED_NPZ_KEYS, _IQL_CKPT_REQUIRED_KEYS,
    run_is_valid, load_completed_runs, existing_quality_complete, append_quality_row,
    resolve_iql_path, compute_stats,
    check_bc_cql_quality_summary_loadable, check_iql_main_summary_valid,
    capture_frozen_file_snapshots, frozen_snapshots_equal, FROZEN_FILES,
    AUDIT_PATH, MANIFEST_PATH, BC_CQL_QUAL_SUM, IQL_MAIN_SUM, PROJECT_ROOT,
)
from scripts.run_envA_v2_iql_sanity import (
    IQL_CFG, train_iql, save_iql_checkpoint,
)
from scripts.run_envA_v2_sanity import (
    encode_obs, OBS_DIM,
)


# ── A. Frozen constants ──────────────────────────────────────────────────────

class TestFrozenConstants:
    def test_seeds_0_to_19(self):
        assert IQL_QUALITY_SEEDS == list(range(20))

    def test_exactly_5_datasets(self):
        assert len(IQL_QUALITY_DATASETS) == 5

    def test_all_5_quality_bins_present(self):
        bins = set(QUALITY_BIN_MAP.values())
        assert bins == {"random", "suboptimal", "medium", "expert", "mixed"}

    def test_algorithm_is_iql(self):
        import importlib, inspect
        src = inspect.getsource(importlib.import_module("scripts.run_envA_v2_iql_quality_sweep"))
        assert '"iql"' in src

    def test_total_runs_100(self):
        assert TOTAL_RUNS == 100

    def test_obs_dim_900(self):
        assert OBS_DIM == 900


# ── B. Pre-flight / schema ───────────────────────────────────────────────────

class TestPreFlightSchema:
    def test_5_quality_datasets_in_manifest(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_QUALITY_DATASETS:
            assert ds in mrows
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            assert os.path.isfile(p)

    def test_11key_schema_real_datasets(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_QUALITY_DATASETS:
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            d = np.load(p, allow_pickle=True)
            for k in REQUIRED_NPZ_KEYS:
                assert k in d, f"{ds} missing '{k}'"

    def test_obs_shape_N2_real_datasets(self):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            mrows = {r["dataset_name"]: r for r in csv.DictReader(f)}
        for ds in IQL_QUALITY_DATASETS:
            p = os.path.join(PROJECT_ROOT, mrows[ds]["file_path"])
            d = np.load(p, allow_pickle=True)
            assert d["observations"].shape[1] == 2, \
                f"{ds} obs shape {d['observations'].shape}, expected (N,2)"

    def test_length_mismatch_detectable(self, tmp_path):
        fake = str(tmp_path / "fake.npz")
        np.savez(fake, observations=np.zeros((10, 2), dtype=np.int32),
                 actions=np.zeros(5))
        d = np.load(fake)
        assert len(d["observations"]) != len(d["actions"])

    def test_wrong_obs_shape_detectable(self, tmp_path):
        fake = str(tmp_path / "fake.npz")
        np.savez(fake, observations=np.zeros((10, 3), dtype=np.int32))
        d = np.load(fake)
        assert d["observations"].shape[1] != 2


# ── C. BC/CQL quality prerequisite helper ───────────────────────────────────

class TestBCCQLQualityHelper:
    def _fake_quality_csv(self, tmp_path, cp_fn, n=200):
        fake = str(tmp_path / "fake_quality.csv")
        cols = ["dataset_name", "quality_bin", "algorithm", "train_seed",
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

    def test_real_bc_cql_quality_spot(self):
        with open(BC_CQL_QUAL_SUM, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 200
        for r in rows[:3]:
            ap = resolve_iql_path(r["checkpoint_path"])
            assert os.path.isfile(ap)
            torch.load(ap, map_location="cpu", weights_only=False)

    def test_helper_detects_missing_file(self, tmp_path):
        fake = self._fake_quality_csv(tmp_path, lambda i: str(tmp_path / f"ghost_{i}.pt"))
        with pytest.raises(AssertionError):
            check_bc_cql_quality_summary_loadable(fake)

    def test_helper_detects_empty_path(self, tmp_path):
        fake = self._fake_quality_csv(tmp_path, lambda i: "")
        with pytest.raises(AssertionError):
            check_bc_cql_quality_summary_loadable(fake)

    def test_helper_detects_corrupt_checkpoint(self, tmp_path):
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"not a checkpoint")
        fake = self._fake_quality_csv(tmp_path, lambda i: corrupt)
        with pytest.raises((AssertionError, Exception)):
            check_bc_cql_quality_summary_loadable(fake)


# ── D. R2 IQL main prerequisite helper ───────────────────────────────────────

class TestIQLMainPrerequisiteHelper:
    def _make_iql_ckpt(self, tmp_path, strip_key=None):
        obs   = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, cfg, 0)
        path  = str(tmp_path / "test_main.pt")
        save_iql_checkpoint(path, actor, q1, q2, v, "ds", 0, cfg, al, ql, vl)
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
            ap = resolve_iql_path(r["checkpoint_path"])
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

    def test_helper_detects_missing_q2_key(self, tmp_path):
        path = self._make_iql_ckpt(tmp_path, strip_key="q2_state_dict")
        fake = self._fake_main_csv(tmp_path, lambda: path)
        with pytest.raises(AssertionError):
            check_iql_main_summary_valid(fake)


# ── E. Resume / verify logic ─────────────────────────────────────────────────

class TestResumeModeLogic:
    def _make_valid_ckpt(self, tmp_path):
        obs   = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, cfg, 0)
        path  = str(tmp_path / "valid_ckpt.pt")
        save_iql_checkpoint(path, actor, q1, q2, v, "ds", 0, cfg, al, ql, vl)
        return path

    def _row(self, cp, status="completed", ret="0.39", sr="1.0"):
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

    def test_existing_quality_complete_false_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.run_envA_v2_iql_quality_sweep.SUMMARY_PATH",
                            str(tmp_path / "nonexistent.csv"))
        ok, _ = existing_quality_complete()
        assert not ok

    def test_existing_quality_complete_false_partial(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "partial.csv")
        monkeypatch.setattr("scripts.run_envA_v2_iql_quality_sweep.SUMMARY_PATH", csv_path)
        for i in range(3):
            row = {c: "" for c in SUMMARY_COLUMNS}
            row.update({"status": "completed", "checkpoint_path": "fake.pt",
                        "algorithm": "iql", "dataset_name": IQL_QUALITY_DATASETS[0],
                        "train_seed": str(i), "avg_return": "0.4", "success_rate": "1.0"})
            append_quality_row(row)
        ok, _ = existing_quality_complete()
        assert not ok

    def test_existing_quality_complete_false_corrupt_ckpt(self, tmp_path, monkeypatch):
        corrupt = str(tmp_path / "corrupt.pt")
        with open(corrupt, "wb") as f:
            f.write(b"garbage")
        csv_path = str(tmp_path / "main.csv")
        rows = []
        for ds in IQL_QUALITY_DATASETS:
            for s in IQL_QUALITY_SEEDS:
                r = {c: "" for c in SUMMARY_COLUMNS}
                r.update({"dataset_name": ds, "algorithm": "iql",
                          "train_seed": str(s), "status": "completed",
                          "checkpoint_path": corrupt,
                          "avg_return": "0.4", "success_rate": "1.0"})
                rows.append(r)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        monkeypatch.setattr("scripts.run_envA_v2_iql_quality_sweep.SUMMARY_PATH", csv_path)
        ok, _ = existing_quality_complete()
        assert not ok

    def test_existing_quality_complete_true_with_valid_ckpts(self, tmp_path, monkeypatch):
        obs   = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, cfg, 0)
        shared = str(tmp_path / "shared.pt")
        save_iql_checkpoint(shared, actor, q1, q2, v, "ds", 0, cfg, al, ql, vl)

        csv_path = str(tmp_path / "complete.csv")
        rows = []
        for ds in IQL_QUALITY_DATASETS:
            for s in IQL_QUALITY_SEEDS:
                r = {c: "" for c in SUMMARY_COLUMNS}
                r.update({"dataset_name": ds, "algorithm": "iql",
                          "train_seed": str(s), "status": "completed",
                          "checkpoint_path": shared,
                          "avg_return": "0.4", "success_rate": "1.0"})
                rows.append(r)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        monkeypatch.setattr("scripts.run_envA_v2_iql_quality_sweep.SUMMARY_PATH", csv_path)
        ok, _ = existing_quality_complete()
        assert ok


# ── F. Checkpoint roundtrip ──────────────────────────────────────────────────

class TestCheckpointRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        obs   = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (200,)).astype(np.int64)
        rews  = np.random.randn(200).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (200, 2)).astype(np.int32))
        terms = (np.random.rand(200) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 3, "batch_size": 32}
        actor, q1, q2, v, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, cfg, 0)
        path  = str(tmp_path / "quality_iql.pt")
        save_iql_checkpoint(path, actor, q1, q2, v, "test_ds", 0, cfg, al, ql, vl)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for k in _IQL_CKPT_REQUIRED_KEYS:
            assert k in ckpt


# ── G. Smoke test ────────────────────────────────────────────────────────────

class TestSmoke:
    def test_iql_smoke_finite_losses(self):
        obs   = encode_obs(np.random.randint(0, 30, (300, 2)).astype(np.int32))
        acts  = np.random.randint(0, 4, (300,)).astype(np.int64)
        rews  = np.random.randn(300).astype(np.float32)
        nobs  = encode_obs(np.random.randint(0, 30, (300, 2)).astype(np.int32))
        terms = (np.random.rand(300) < 0.05).astype(np.float32)
        cfg   = {**IQL_CFG, "num_updates": 10, "batch_size": 32}
        _, _, _, _, al, ql, vl = train_iql(obs, acts, rews, nobs, terms, cfg, 0)
        assert math.isfinite(al) and math.isfinite(ql) and math.isfinite(vl)

    def test_iql_cfg_inherited_from_r1(self):
        from scripts.run_envA_v2_iql_sanity import IQL_CFG as R1_CFG
        from scripts.run_envA_v2_iql_quality_sweep import IQL_CFG as R4_CFG
        assert R4_CFG is R1_CFG


# ── H. Frozen snapshot helpers ───────────────────────────────────────────────

class TestFrozenFileSnapshot:
    def test_unchanged_files_compare_equal(self, tmp_path):
        f1 = tmp_path / "frozen1.csv"
        f2 = tmp_path / "frozen2.csv"
        f1.write_text("header\nrow1\n")
        f2.write_text("header\nrowA\n")
        paths = [str(f1), str(f2)]
        before = capture_frozen_file_snapshots(paths)
        after  = capture_frozen_file_snapshots(paths)
        ok, diffs = frozen_snapshots_equal(before, after)
        assert ok, f"Expected no diffs, got: {diffs}"

    def test_modified_content_detected(self, tmp_path):
        f = tmp_path / "frozen.csv"
        f.write_text("original\n")
        paths = [str(f)]
        before = capture_frozen_file_snapshots(paths)
        f.write_text("MODIFIED\n")
        after = capture_frozen_file_snapshots(paths)
        ok, diffs = frozen_snapshots_equal(before, after)
        assert not ok

    def test_deleted_file_detected(self, tmp_path):
        f = tmp_path / "frozen.csv"
        f.write_text("content\n")
        paths = [str(f)]
        before = capture_frozen_file_snapshots(paths)
        f.unlink()
        after = capture_frozen_file_snapshots(paths)
        ok, diffs = frozen_snapshots_equal(before, after)
        assert not ok

    def test_gate_not_hardcoded_true(self):
        import importlib, inspect
        src = inspect.getsource(importlib.import_module("scripts.run_envA_v2_iql_quality_sweep"))
        assert 'gate["no_frozen_files_modified"] = True' not in src
        assert "frozen_snapshots_equal" in src


# ── I. No scope creep ────────────────────────────────────────────────────────

class TestNoScopeCreep:
    def test_no_manifest_write(self):
        import importlib, inspect
        src = inspect.getsource(importlib.import_module("scripts.run_envA_v2_iql_quality_sweep"))
        assert 'open(MANIFEST_PATH, "w"' not in src

    def test_no_audit_write(self):
        import importlib, inspect
        src = inspect.getsource(importlib.import_module("scripts.run_envA_v2_iql_quality_sweep"))
        assert 'open(AUDIT_PATH, "w"' not in src

    def test_no_old_chain(self):
        import importlib, inspect
        src = inspect.getsource(importlib.import_module("scripts.run_envA_v2_iql_quality_sweep"))
        assert "EnvA_main_rebuild" not in src
        assert "run_hopper_benchmark" not in src

    def test_no_behavior_catalog_write(self):
        import importlib, inspect
        src = inspect.getsource(importlib.import_module("scripts.run_envA_v2_iql_quality_sweep"))
        assert "behavior_policy_catalog" not in src
