"""
tests/test_envA_v2_quality_sweep.py
Clean Phase 10: verify quality sweep frozen rules, config inheritance, minimal runnability.
"""

import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from scripts.run_envA_v2_quality_sweep import (
    QUALITY_SEEDS, QUALITY_DATASETS, ALGORITHMS,
    QUALITY_BIN_MAP, SUMMARY_COLUMNS,
    QUALITY_DIR, QUALITY_SUMMARY,
    AUDIT_PATH, ENVA_MAIN_SUMMARY, VAL_SUMMARY,
    existing_quality_summary_complete,
    check_summary_loadable, resolve_qpath,
)
from scripts.run_envA_v2_sanity import (
    BC_CFG as SAN_BC_CFG, CQL_CFG as SAN_CQL_CFG,
    MLP, train_bc, train_cql, load_dataset,
    OBS_DIM, N_ACTIONS,
)
from scripts.run_envA_v2_quality_sweep import (
    BC_CFG as Q_BC_CFG, CQL_CFG as Q_CQL_CFG,
)


# ── 1. Frozen constants ──────────────────────────────────────────────────────

class TestFrozenConstants:
    def test_seeds_0_to_19(self):
        assert QUALITY_SEEDS == list(range(20))

    def test_5_quality_datasets(self):
        assert len(QUALITY_DATASETS) == 5
        assert "envA_v2_quality_random_wide50k" in QUALITY_DATASETS
        assert "envA_v2_quality_mixed_wide50k" in QUALITY_DATASETS

    def test_algorithms(self):
        assert ALGORITHMS == ["bc", "cql"]

    def test_quality_bin_map(self):
        assert QUALITY_BIN_MAP["envA_v2_quality_random_wide50k"] == "random"
        assert QUALITY_BIN_MAP["envA_v2_quality_expert_wide50k"] == "expert"


# ── 2. Config inheritance ────────────────────────────────────────────────────

class TestConfigInheritance:
    def test_bc_cfg_same_object(self):
        assert Q_BC_CFG is SAN_BC_CFG

    def test_cql_cfg_same_object(self):
        assert Q_CQL_CFG is SAN_CQL_CFG

    def test_bc_values(self):
        assert Q_BC_CFG["num_updates"] == 5000
        assert Q_BC_CFG["hidden_dims"] == [256, 256]

    def test_cql_values(self):
        assert Q_CQL_CFG["cql_alpha"] == 1.0
        assert Q_CQL_CFG["gamma"] == 0.99


# ── 3. Summary schema ────────────────────────────────────────────────────────

class TestSummarySchema:
    def test_columns_count(self):
        assert len(SUMMARY_COLUMNS) == 13

    def test_has_quality_bin(self):
        assert "quality_bin" in SUMMARY_COLUMNS

    def test_has_checkpoint_path(self):
        assert "checkpoint_path" in SUMMARY_COLUMNS


# ── 4. Pre-flight conditions ─────────────────────────────────────────────────

class TestPreFlight:
    def test_audit_frozen(self):
        with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert all(r["freeze_ready"] == "yes" for r in rows)

    def test_main_completed(self):
        with open(ENVA_MAIN_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        assert all(r["status"] == "completed" for r in rows)

    def test_validation_completed(self):
        with open(VAL_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        assert all(r["status"] == "completed" for r in rows)

    def test_5_quality_datasets_loadable(self):
        for dn in QUALITY_DATASETS:
            obs, acts, rews, nobs, terms = load_dataset(dn)
            assert obs.shape[1] == OBS_DIM
            assert len(acts) == len(obs)


# ── 5. Training smoke ────────────────────────────────────────────────────────

class TestTrainingSmoke:
    @pytest.fixture(scope="class")
    def small_data(self):
        obs, acts, rews, nobs, terms = load_dataset("envA_v2_quality_random_wide50k")
        n = 500
        return obs[:n], acts[:n], rews[:n], nobs[:n], terms[:n]

    def test_bc_mini(self, small_data):
        obs, acts, _, _, _ = small_data
        cfg = {**Q_BC_CFG, "num_updates": 5, "batch_size": 32}
        model, loss = train_bc(obs, acts, cfg, 0)
        assert math.isfinite(loss)

    def test_cql_mini(self, small_data):
        obs, acts, rews, nobs, terms = small_data
        cfg = {**Q_CQL_CFG, "num_updates": 5, "batch_size": 32}
        model, loss = train_cql(obs, acts, rews, nobs, terms, cfg, 0)
        assert math.isfinite(loss)


# ── 6. Checkpoint roundtrip ──────────────────────────────────────────────────

class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        m = MLP(OBS_DIM, N_ACTIONS, [64])
        p = str(tmp_path / "test.pt")
        torch.save({"model_state_dict": m.state_dict()}, p)
        loaded = torch.load(p, weights_only=False)
        m2 = MLP(OBS_DIM, N_ACTIONS, [64])
        m2.load_state_dict(loaded["model_state_dict"])
        x = torch.randn(1, OBS_DIM)
        assert torch.allclose(m(x), m2(x))


# ── 7. Patch A: main summary checkpoint loadability ──────────────────────────

class TestMainSummaryLoadability:

    def test_main_summary_loadable(self):
        """First 2 rows of real main summary must pass check_summary_loadable."""
        with open(ENVA_MAIN_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Spot-check first 2 for speed
        for r in rows[:2]:
            cp = r.get("checkpoint_path", "").strip()
            assert cp
            assert os.path.isfile(resolve_qpath(cp))
            torch.load(resolve_qpath(cp), weights_only=False)

    def test_missing_main_ckpt_fails(self, tmp_path):
        fake_csv = str(tmp_path / "fake_main.csv")
        with open(fake_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset_name","algorithm","train_seed","num_updates",
                         "final_train_loss","eval_episodes","avg_return",
                         "success_rate","avg_episode_length","checkpoint_path",
                         "status","notes"])
            w.writerow(["d","bc","0","5000","0.5","100","0.3","1.0","60",
                         "nonexistent/bad.pt","completed","ok"])
        with pytest.raises(AssertionError):
            check_summary_loadable(fake_csv, 1)


# ── 8. Patch B: validation summary checkpoint loadability ─────────────────────

class TestValSummaryLoadability:

    def test_val_summary_loadable(self):
        """First 2 rows of real validation summary must pass."""
        with open(VAL_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows[:2]:
            cp = r.get("checkpoint_path", "").strip()
            assert cp
            assert os.path.isfile(resolve_qpath(cp))
            torch.load(resolve_qpath(cp), weights_only=False)

    def test_missing_val_ckpt_fails(self, tmp_path):
        fake_csv = str(tmp_path / "fake_val.csv")
        with open(fake_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["env_name","dataset_name","algorithm","train_seed","obs_dim",
                         "num_updates","final_train_loss","eval_episodes","avg_return",
                         "success_rate","avg_episode_length","checkpoint_path",
                         "status","notes"])
            w.writerow(["EnvB","d","bc","0","225","5000","0.5","100","0.3","1.0","60",
                         "nonexistent/bad.pt","completed","ok"])
        with pytest.raises(AssertionError):
            check_summary_loadable(fake_csv, 1)


# ── 9. Patch C: read-only constraint ─────────────────────────────────────────

class TestReadOnlyConstraint:

    def test_no_manifest_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_quality_sweep")
        src = inspect.getsource(mod)
        assert 'open(MANIFEST_PATH, "w"' not in src
        assert "MANIFEST_PATH, 'w'" not in src

    def test_no_catalog_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_quality_sweep")
        src = inspect.getsource(mod)
        # Script doesn't even import CATALOG_PATH, so no write possible
        assert 'CATALOG_PATH' not in src or 'open(CATALOG_PATH, "w"' not in src

    def test_no_audit_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_quality_sweep")
        src = inspect.getsource(mod)
        assert 'open(AUDIT_PATH, "w"' not in src


# ── 10. Patch D: reuse mode ──────────────────────────────────────────────────

class TestReuseMode:

    def test_complete_summary_detected(self):
        ok, rows = existing_quality_summary_complete()
        assert ok, f"Expected complete summary, got {len(rows)} rows"
        assert len(rows) == 200

    def test_incomplete_not_detected(self, tmp_path):
        fake = str(tmp_path / "fake_q.csv")
        with open(fake, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            for i in range(3):
                w.writerow({c: "x" for c in SUMMARY_COLUMNS})
        import scripts.run_envA_v2_quality_sweep as mod
        orig = mod.QUALITY_SUMMARY
        mod.QUALITY_SUMMARY = fake
        try:
            ok, _ = existing_quality_summary_complete()
            assert not ok
        finally:
            mod.QUALITY_SUMMARY = orig


# ── 11. Patch E: quality summary checkpoint gate ──────────────────────────────

class TestQualitySummaryGate:

    def test_quality_ckpts_exist(self):
        """First 2 rows of quality summary have existing checkpoint files."""
        with open(QUALITY_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows[:2]:
            cp = r.get("checkpoint_path", "").strip()
            assert cp
            assert os.path.isfile(resolve_qpath(cp))

    def test_quality_ckpts_loadable(self):
        """First 2 rows of quality summary have loadable checkpoints."""
        with open(QUALITY_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows[:2]:
            cp = r.get("checkpoint_path", "").strip()
            torch.load(resolve_qpath(cp), weights_only=False)


# ── 12. No old-chain imports ─────────────────────────────────────────────────

class TestNoOldChain:
    def test_no_old_imports(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_quality_sweep")
        src = inspect.getsource(mod)
        assert "EnvA_main_rebuild" not in src
        assert "train_behavior_pool" not in src
        assert "benchmark" not in src.lower()
