"""
tests/test_envbc_validation.py
Clean Phase 9: verify EnvB/C validation frozen rules, encoders, minimal runnability.
"""

import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from scripts.run_envbc_validation import (
    VALIDATION_SEEDS, VALIDATION_DATASETS, ALGORITHMS,
    ENVB_OBS_DIM, ENVC_OBS_DIM, SUMMARY_COLUMNS,
    AUDIT_PATH, ENVA_MAIN_SUMMARY, VAL_SUMMARY,
    encode_envB, encode_envC, encode_single_B, encode_single_C,
    load_validation_dataset,
    train_bc, train_cql, evaluate,
    MLP, BC_CFG, CQL_CFG, N_ACTIONS, REQUIRED_KEYS,
    resolve_val_path, check_main_ckpts_loadable, check_dataset_schema,
    existing_val_summary_is_complete,
)
from scripts.run_envA_v2_sanity import (
    BC_CFG as SAN_BC_CFG, CQL_CFG as SAN_CQL_CFG,
)


# ── 1. Frozen constants ──────────────────────────────────────────────────────

class TestFrozenConstants:
    def test_seeds_0_to_19(self):
        assert VALIDATION_SEEDS == list(range(20))

    def test_datasets(self):
        assert VALIDATION_DATASETS == [
            "envB_small_wide_medium", "envB_large_narrow_medium",
            "envC_small_wide_medium", "envC_large_narrow_medium",
        ]

    def test_algorithms(self):
        assert ALGORITHMS == ["bc", "cql"]

    def test_envb_obs_dim(self):
        assert ENVB_OBS_DIM == 225

    def test_envc_obs_dim(self):
        assert ENVC_OBS_DIM == 450

    def test_bc_cfg_values_match_sanity(self):
        assert BC_CFG["num_updates"] == SAN_BC_CFG["num_updates"]
        assert BC_CFG["hidden_dims"] == SAN_BC_CFG["hidden_dims"]
        assert BC_CFG["lr"] == SAN_BC_CFG["lr"]

    def test_cql_cfg_values_match_sanity(self):
        assert CQL_CFG["num_updates"] == SAN_CQL_CFG["num_updates"]
        assert CQL_CFG["cql_alpha"] == SAN_CQL_CFG["cql_alpha"]
        assert CQL_CFG["gamma"] == SAN_CQL_CFG["gamma"]


# ── 2. Encoders ──────────────────────────────────────────────────────────────

class TestEncoders:
    def test_envB_encode_shape(self):
        obs = np.array([[1, 1], [5, 5]], dtype=np.int32)
        enc = encode_envB(obs)
        assert enc.shape == (2, 225)
        assert enc.dtype == np.float32

    def test_envC_encode_shape(self):
        obs = np.array([[1, 1, 0], [5, 5, 1]], dtype=np.int32)
        enc = encode_envC(obs)
        assert enc.shape == (2, 450)
        assert enc.dtype == np.float32

    def test_envB_single(self):
        t = encode_single_B(1, 1)
        assert t.shape == (225,)
        assert t.sum().item() == 1.0

    def test_envC_single(self):
        t = encode_single_C(((1, 1), 0))
        assert t.shape == (450,)
        assert t.sum().item() == 1.0


# ── 3. Data loading ──────────────────────────────────────────────────────────

class TestDataLoading:
    def test_envB_small_wide(self):
        obs, acts, rews, nobs, terms, env, dim = load_validation_dataset("envB_small_wide_medium")
        assert obs.shape[1] == 225
        assert env == "EnvB"
        assert dim == 225

    def test_envC_small_wide(self):
        obs, acts, rews, nobs, terms, env, dim = load_validation_dataset("envC_small_wide_medium")
        assert obs.shape[1] == 450
        assert env == "EnvC"
        assert dim == 450


# ── 4. Model shapes ──────────────────────────────────────────────────────────

class TestModelShapes:
    def test_bc_envB(self):
        m = MLP(225, N_ACTIONS, BC_CFG["hidden_dims"])
        assert m(torch.randn(2, 225)).shape == (2, 4)

    def test_cql_envC(self):
        m = MLP(450, N_ACTIONS, CQL_CFG["hidden_dims"])
        assert m(torch.randn(2, 450)).shape == (2, 4)


# ── 5. Training smoke ────────────────────────────────────────────────────────

class TestTrainingSmoke:
    @pytest.fixture(scope="class")
    def envB_data(self):
        obs, acts, rews, nobs, terms, _, _ = load_validation_dataset("envB_small_wide_medium")
        n = 500
        return obs[:n], acts[:n], rews[:n], nobs[:n], terms[:n]

    @pytest.fixture(scope="class")
    def envC_data(self):
        obs, acts, rews, nobs, terms, _, _ = load_validation_dataset("envC_small_wide_medium")
        n = 500
        return obs[:n], acts[:n], rews[:n], nobs[:n], terms[:n]

    def test_bc_envB_runs(self, envB_data):
        obs, acts, _, _, _ = envB_data
        cfg = {**BC_CFG, "num_updates": 5, "batch_size": 32}
        model, loss = train_bc(obs, acts, cfg, 0, 225)
        assert math.isfinite(loss)

    def test_cql_envC_runs(self, envC_data):
        obs, acts, rews, nobs, terms = envC_data
        cfg = {**CQL_CFG, "num_updates": 5, "batch_size": 32}
        model, loss = train_cql(obs, acts, rews, nobs, terms, cfg, 0, 450)
        assert math.isfinite(loss)


# ── 6. Checkpoint roundtrip ──────────────────────────────────────────────────

class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        m = MLP(225, 4, [64])
        p = str(tmp_path / "test.pt")
        torch.save({"model_state_dict": m.state_dict()}, p)
        loaded = torch.load(p, weights_only=False)
        m2 = MLP(225, 4, [64])
        m2.load_state_dict(loaded["model_state_dict"])
        x = torch.randn(1, 225)
        assert torch.allclose(m(x), m2(x))


# ── 7. Summary schema ────────────────────────────────────────────────────────

class TestSummarySchema:
    def test_columns_count(self):
        assert len(SUMMARY_COLUMNS) == 14

    def test_has_env_name(self):
        assert "env_name" in SUMMARY_COLUMNS

    def test_has_obs_dim(self):
        assert "obs_dim" in SUMMARY_COLUMNS


# ── 8. Pre-flight conditions ─────────────────────────────────────────────────

class TestPreFlight:
    def test_audit_frozen(self):
        with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert all(r["freeze_ready"] == "yes" for r in rows)

    def test_main_experiment_completed(self):
        with open(ENVA_MAIN_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        assert all(r["status"] == "completed" for r in rows)


# ── 9. Patch: main checkpoint loadability pre-flight ──────────────────────────

class TestMainCkptLoadability:

    def test_valid_ckpts_pass(self):
        with open(ENVA_MAIN_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Just check first 2 for speed
        ok, failed = check_main_ckpts_loadable(rows[:2])
        assert ok, f"Main ckpts not loadable: {failed}"

    def test_missing_ckpt_fails(self):
        fake = [{"checkpoint_path": "nonexistent/bad.pt", "dataset_name": "d"}]
        ok, _ = check_main_ckpts_loadable(fake)
        assert not ok

    def test_empty_path_fails(self):
        fake = [{"checkpoint_path": "", "dataset_name": "d"}]
        ok, _ = check_main_ckpts_loadable(fake)
        assert not ok


# ── 10. Patch: dataset schema pre-flight ──────────────────────────────────────

class TestDatasetSchema:

    def test_envB_schema_ok(self):
        ok, notes = check_dataset_schema("envB_small_wide_medium", "EnvB")
        assert ok, f"Schema fail: {notes}"

    def test_envC_schema_ok(self):
        ok, notes = check_dataset_schema("envC_small_wide_medium", "EnvC")
        assert ok, f"Schema fail: {notes}"

    def test_wrong_obs_shape_fails(self, tmp_path):
        # Fake EnvB dataset with (N,3) instead of (N,2)
        n = 10
        fpath = str(tmp_path / "bad_envB.npz")
        np.savez(fpath,
                 observations=np.zeros((n, 3), dtype=np.int32),  # wrong!
                 actions=np.zeros(n, dtype=np.int32),
                 rewards=np.zeros(n, dtype=np.float32),
                 next_observations=np.zeros((n, 3), dtype=np.int32),
                 terminals=np.zeros(n, dtype=bool),
                 truncations=np.zeros(n, dtype=bool),
                 episode_ids=np.zeros(n, dtype=np.int32),
                 timesteps=np.arange(n, dtype=np.int32),
                 source_policy_ids=np.array(["x"] * n, dtype=object),
                 source_train_seeds=np.zeros(n, dtype=np.int32),
                 source_behavior_epsilons=np.zeros(n, dtype=np.float32))
        # Monkeypatch manifest to point to this fake file
        import scripts.run_envbc_validation as mod
        orig = mod.MANIFEST_PATH
        fake_manifest = str(tmp_path / "manifest.csv")
        with open(fake_manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset_name", "env_name", "file_path"])
            w.writeheader()
            w.writerow({"dataset_name": "fake_envB", "env_name": "EnvB", "file_path": fpath})
        mod.MANIFEST_PATH = fake_manifest
        try:
            ok, notes = check_dataset_schema("fake_envB", "EnvB")
            assert not ok
            assert any("obs shape" in n for n in notes)
        finally:
            mod.MANIFEST_PATH = orig

    def test_missing_keys_fails(self, tmp_path):
        n = 5
        fpath = str(tmp_path / "incomplete.npz")
        np.savez(fpath, observations=np.zeros((n, 2), dtype=np.int32))  # only 1 key
        import scripts.run_envbc_validation as mod
        orig = mod.MANIFEST_PATH
        fake_manifest = str(tmp_path / "manifest2.csv")
        with open(fake_manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset_name", "env_name", "file_path"])
            w.writeheader()
            w.writerow({"dataset_name": "inc", "env_name": "EnvB", "file_path": fpath})
        mod.MANIFEST_PATH = fake_manifest
        try:
            ok, notes = check_dataset_schema("inc", "EnvB")
            assert not ok
            assert any("missing keys" in n for n in notes)
        finally:
            mod.MANIFEST_PATH = orig


# ── 11. Patch: reuse mode ─────────────────────────────────────────────────────

class TestReuseMode:

    def test_complete_summary_detected(self):
        ok, rows = existing_val_summary_is_complete()
        assert ok, f"Expected complete summary, got {len(rows)} rows"
        assert len(rows) == 160

    def test_incomplete_not_detected(self, tmp_path):
        fake_path = str(tmp_path / "fake.csv")
        with open(fake_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            for i in range(5):
                w.writerow({c: "x" for c in SUMMARY_COLUMNS})
        import scripts.run_envbc_validation as mod
        orig = mod.VAL_SUMMARY
        mod.VAL_SUMMARY = fake_path
        try:
            ok, _ = existing_val_summary_is_complete()
            assert not ok
        finally:
            mod.VAL_SUMMARY = orig


# ── 12. Patch: path normalization ─────────────────────────────────────────────

class TestPathNormalization:

    def test_backslash(self):
        p = resolve_val_path("artifacts\\training_validation\\test.pt")
        assert "training_validation" in p

    def test_forward_slash(self):
        p = resolve_val_path("artifacts/training_validation/test.pt")
        assert "training_validation" in p


# ── 13. No old-chain imports ─────────────────────────────────────────────────

class TestNoOldChain:
    def test_no_old_imports(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envbc_validation")
        src = inspect.getsource(mod)
        assert "EnvA_main_rebuild" not in src
        assert "train_behavior_pool" not in src
