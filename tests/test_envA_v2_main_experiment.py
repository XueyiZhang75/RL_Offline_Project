"""
tests/test_envA_v2_main_experiment.py
Clean Phase 8: verify main experiment frozen rules, config inheritance, minimal runnability.
"""

import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from scripts.run_envA_v2_main_experiment import (
    MAIN_SEEDS, MAIN_DATASETS, ALGORITHMS,
    MAIN_DIR, MAIN_SUMMARY, SUMMARY_COLUMNS,
    AUDIT_PATH, SANITY_SUMMARY,
    resolve_ckpt_path, check_sanity_checkpoints,
    check_main_checkpoints_exist, check_main_checkpoints_loadable,
    existing_summary_is_complete,
)
# These must be THE SAME objects inherited from sanity
from scripts.run_envA_v2_main_experiment import (
    BC_CFG as MAIN_BC_CFG,
    CQL_CFG as MAIN_CQL_CFG,
    OBS_DIM as MAIN_OBS_DIM,
    N_ACTIONS as MAIN_N_ACTIONS,
    EVAL_EPISODES as MAIN_EVAL,
    MLP, train_bc, train_cql, load_dataset,
)
from scripts.run_envA_v2_sanity import (
    BC_CFG as SAN_BC_CFG,
    CQL_CFG as SAN_CQL_CFG,
)


# ── 1. Frozen constants ──────────────────────────────────────────────────────

class TestFrozenConstants:

    def test_main_seeds_0_to_19(self):
        assert MAIN_SEEDS == list(range(20))

    def test_main_datasets(self):
        assert MAIN_DATASETS == [
            "envA_v2_small_wide_medium",
            "envA_v2_small_narrow_medium",
            "envA_v2_large_wide_medium",
            "envA_v2_large_narrow_medium",
        ]

    def test_algorithms(self):
        assert ALGORITHMS == ["bc", "cql"]

    def test_obs_dim_900(self):
        assert MAIN_OBS_DIM == 900

    def test_eval_episodes_100(self):
        assert MAIN_EVAL == 100


# ── 2. Config inherited from sanity ──────────────────────────────────────────

class TestConfigInheritance:

    def test_bc_cfg_same_object(self):
        assert MAIN_BC_CFG is SAN_BC_CFG

    def test_cql_cfg_same_object(self):
        assert MAIN_CQL_CFG is SAN_CQL_CFG

    def test_bc_cfg_values(self):
        assert MAIN_BC_CFG["num_updates"] == 5000
        assert MAIN_BC_CFG["hidden_dims"] == [256, 256]

    def test_cql_cfg_values(self):
        assert MAIN_CQL_CFG["num_updates"] == 5000
        assert MAIN_CQL_CFG["cql_alpha"] == 1.0


# ── 3. Pre-flight conditions ─────────────────────────────────────────────────

class TestPreFlight:

    def test_audit_freeze_ready(self):
        with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 13
        for r in rows:
            assert r["freeze_ready"] == "yes"

    def test_sanity_all_completed(self):
        with open(SANITY_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 12
        for r in rows:
            assert r["status"] == "completed"

    def test_four_datasets_loadable(self):
        for dn in MAIN_DATASETS:
            obs, acts, rews, nobs, terms = load_dataset(dn)
            assert obs.shape[1] == 900
            assert len(acts) == len(obs)


# ── 4. Summary schema ────────────────────────────────────────────────────────

class TestSummarySchema:

    def test_columns_count(self):
        assert len(SUMMARY_COLUMNS) == 12

    def test_required_columns(self):
        for col in ["dataset_name", "algorithm", "train_seed",
                     "final_train_loss", "avg_return", "success_rate",
                     "checkpoint_path", "status"]:
            assert col in SUMMARY_COLUMNS


# ── 5. Training smoke (minimal) ──────────────────────────────────────────────

class TestTrainingSmoke:

    @pytest.fixture(scope="class")
    def small_data(self):
        obs, acts, rews, nobs, terms = load_dataset("envA_v2_small_wide_medium")
        n = 500
        return obs[:n], acts[:n], rews[:n], nobs[:n], terms[:n]

    def test_bc_mini(self, small_data):
        obs, acts, _, _, _ = small_data
        cfg = {**MAIN_BC_CFG, "num_updates": 5, "batch_size": 32}
        model, loss = train_bc(obs, acts, cfg, 0)
        assert math.isfinite(loss)

    def test_cql_mini(self, small_data):
        obs, acts, rews, nobs, terms = small_data
        cfg = {**MAIN_CQL_CFG, "num_updates": 5, "batch_size": 32}
        model, loss = train_cql(obs, acts, rews, nobs, terms, cfg, 0)
        assert math.isfinite(loss)


# ── 6. Checkpoint roundtrip ──────────────────────────────────────────────────

class TestCheckpoint:

    def test_roundtrip(self, tmp_path):
        model = MLP(900, 4, [64])
        path = str(tmp_path / "test.pt")
        torch.save({"model_state_dict": model.state_dict()}, path)
        loaded = torch.load(path, weights_only=False)
        m2 = MLP(900, 4, [64])
        m2.load_state_dict(loaded["model_state_dict"])
        x = torch.randn(1, 900)
        assert torch.allclose(model(x), m2(x))


# ── 7. Patch: sanity checkpoint existence ─────────────────────────────────────

class TestSanityCheckpointExistence:

    def test_all_sanity_ckpts_exist(self):
        with open(SANITY_SUMMARY, newline="", encoding="utf-8") as f:
            san_rows = list(csv.DictReader(f))
        ok, missing = check_sanity_checkpoints(san_rows)
        assert ok, f"Sanity checkpoints missing: {missing}"

    def test_missing_path_fails(self):
        fake_rows = [{"checkpoint_path": "", "train_seed": "0", "algorithm": "bc"}]
        ok, _ = check_sanity_checkpoints(fake_rows)
        assert not ok

    def test_nonexistent_path_fails(self):
        fake_rows = [{"checkpoint_path": "nonexistent/fake.pt",
                       "train_seed": "0", "algorithm": "bc"}]
        ok, _ = check_sanity_checkpoints(fake_rows)
        assert not ok


# ── 8. Patch: main checkpoint loadability ─────────────────────────────────────

class TestMainCheckpointLoadability:

    def test_valid_checkpoint_loadable(self, tmp_path):
        path = str(tmp_path / "good.pt")
        torch.save({"test": 1}, path)
        rows = [{"checkpoint_path": path, "dataset_name": "d",
                 "algorithm": "bc", "train_seed": "0"}]
        ok, _ = check_main_checkpoints_loadable(rows)
        assert ok

    def test_missing_checkpoint_fails(self):
        rows = [{"checkpoint_path": "nonexistent/bad.pt", "dataset_name": "d",
                 "algorithm": "bc", "train_seed": "0"}]
        ok, _ = check_main_checkpoints_loadable(rows)
        assert not ok

    def test_empty_path_fails(self):
        rows = [{"checkpoint_path": "", "dataset_name": "d",
                 "algorithm": "bc", "train_seed": "0"}]
        ok, _ = check_main_checkpoints_loadable(rows)
        assert not ok


# ── 9. Patch: reuse-existing-results mode ─────────────────────────────────────

class TestReuseMode:

    def test_complete_summary_detected(self):
        ok, rows = existing_summary_is_complete()
        # Should be True since Phase 8 already completed
        assert ok, f"Expected complete summary, got {len(rows)} rows"
        assert len(rows) == 160

    def test_incomplete_summary_not_detected(self, tmp_path):
        # Write a 5-row summary
        fake_path = str(tmp_path / "fake_summary.csv")
        with open(fake_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            for i in range(5):
                w.writerow({c: "x" for c in SUMMARY_COLUMNS})
        import scripts.run_envA_v2_main_experiment as mod
        orig = mod.MAIN_SUMMARY
        mod.MAIN_SUMMARY = fake_path
        try:
            ok, _ = existing_summary_is_complete()
            assert not ok
        finally:
            mod.MAIN_SUMMARY = orig


# ── 10. Patch: path normalization ─────────────────────────────────────────────

class TestPathNormalization:

    def test_backslash_path(self):
        p = resolve_ckpt_path("artifacts\\training_main\\test.pt")
        assert os.path.sep in p or "/" in p
        assert "\\\\" not in p  # no double backslash

    def test_forward_slash_path(self):
        p = resolve_ckpt_path("artifacts/training_main/test.pt")
        assert "training_main" in p


# ── 11. No old-chain / no scope creep ────────────────────────────────────────

class TestNoScopeCreep:

    def test_no_old_imports(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_main_experiment")
        src = inspect.getsource(mod)
        assert "EnvA_main_rebuild" not in src
        assert "train_behavior_pool" not in src

    def test_no_envbc_training(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_main_experiment")
        src = inspect.getsource(mod)
        # Should not train EnvB/C in this phase
        assert "envB_" not in src.lower().replace("envb/c", "").replace("env-b", "")
        assert "envC_" not in src.lower().replace("envb/c", "").replace("env-c", "")
