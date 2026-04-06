"""
tests/test_envA_v2_mechanism_analysis.py
Clean Phase 11: verify mechanism analysis frozen rules, metric definitions, minimal runnability.
"""

import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch

from scripts.run_envA_v2_mechanism_analysis import (
    ENVA_V2_DATASETS, SEED_COLUMNS, SUMMARY_COLUMNS,
    AUDIT_PATH, MAIN_SUMMARY, QUALITY_SUMMARY,
    SEED_CSV, SUMMARY_CSV, ANALYSIS_DIR, MECH_EVAL_EPISODES,
    build_dataset_support, mechanism_eval, resolve_path,
)
from scripts.run_envA_v2_sanity import (
    MLP, encode_single, OBS_DIM, N_ACTIONS,
)


# ── 1. Frozen constants ──────────────────────────────────────────────────────

class TestFrozenConstants:
    def test_9_datasets(self):
        assert len(ENVA_V2_DATASETS) == 9

    def test_datasets_all_envA_v2(self):
        for dn in ENVA_V2_DATASETS:
            assert dn.startswith("envA_v2_")

    def test_eval_episodes_100(self):
        assert MECH_EVAL_EPISODES == 100


# ── 2. CSV schemas ───────────────────────────────────────────────────────────

class TestSchemas:
    def test_seed_columns(self):
        assert "ood_action_rate_step" in SEED_COLUMNS
        assert "support_overlap_rate_unique_sa" in SEED_COLUMNS
        assert "q_over_rate" in SEED_COLUMNS
        assert "checkpoint_path" in SEED_COLUMNS

    def test_summary_columns(self):
        assert "mean_ood_action_rate_step" in SUMMARY_COLUMNS
        assert "mean_q_over_rate" in SUMMARY_COLUMNS
        assert "ci95_run_return_low" in SUMMARY_COLUMNS
        assert "n_runs" in SUMMARY_COLUMNS


# ── 3. Pre-flight conditions ─────────────────────────────────────────────────

class TestPreFlight:
    def test_audit_frozen(self):
        with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert all(r["freeze_ready"] == "yes" for r in rows)

    def test_main_summary(self):
        with open(MAIN_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        assert all(r["status"] == "completed" for r in rows)

    def test_quality_summary(self):
        with open(QUALITY_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 200
        assert all(r["status"] == "completed" for r in rows)


# ── 4. Dataset support construction ──────────────────────────────────────────

class TestDatasetSupport:
    def test_support_nonempty(self):
        sa, metrics = build_dataset_support("envA_v2_small_wide_medium")
        assert len(sa) > 0
        assert metrics["norm_sa_cov"] > 0

    def test_support_sa_tuples(self):
        sa, _ = build_dataset_support("envA_v2_small_wide_medium")
        sample = next(iter(sa))
        assert len(sample) == 3  # (row, col, action)


# ── 5. OOD-action rate definition ────────────────────────────────────────────

class TestOODRate:
    def test_fully_in_support(self):
        """If eval always picks actions in support, OOD rate = 0."""
        # Build a model that always picks action 0
        model = MLP(OBS_DIM, N_ACTIONS, [32])
        # Make huge support covering action 0 for all states
        support_sa = set()
        for r in range(30):
            for c in range(30):
                support_sa.add((r, c, 0))
        # Force model to always pick action 0
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.zero_()
            model.net[-1].bias[0] = 100.0  # action 0 dominates
        mech = mechanism_eval(model, support_sa, "bc", "test", {"return_p90": 0.5})
        assert mech["ood_action_rate_step"] == 0.0

    def test_fully_out_of_support(self):
        """If support is empty, OOD rate = 1."""
        model = MLP(OBS_DIM, N_ACTIONS, [32])
        mech = mechanism_eval(model, set(), "bc", "test", {"return_p90": 0.5})
        assert mech["ood_action_rate_step"] == 1.0


# ── 6. Support overlap & distance ────────────────────────────────────────────

class TestOverlapAndDistance:
    def test_overlap_plus_distance_equals_1(self):
        model = MLP(OBS_DIM, N_ACTIONS, [32])
        sa, metrics = build_dataset_support("envA_v2_small_wide_medium")
        mech = mechanism_eval(model, sa, "bc", "test", metrics)
        overlap = mech["support_overlap_rate_unique_sa"]
        dist = mech["support_distance_proxy"]
        assert abs(overlap + dist - 1.0) < 1e-9


# ── 7. Q-value overestimation proxy ──────────────────────────────────────────

class TestQOverestimation:
    def test_bc_returns_empty(self):
        model = MLP(OBS_DIM, N_ACTIONS, [32])
        sa, metrics = build_dataset_support("envA_v2_small_wide_medium")
        mech = mechanism_eval(model, sa, "bc", "test", metrics)
        assert mech["q_over_rate"] in ("", "NA")
        assert mech["q_over_mean_excess"] in ("", "NA")

    def test_cql_returns_numbers(self):
        model = MLP(OBS_DIM, N_ACTIONS, [32])
        sa, metrics = build_dataset_support("envA_v2_small_wide_medium")
        mech = mechanism_eval(model, sa, "cql", "test", metrics)
        assert mech["q_over_rate"] not in ("", "NA")
        assert math.isfinite(float(mech["q_over_rate"]))
        assert math.isfinite(float(mech["q_over_mean_excess"]))


# ── 8. Smoke test with real checkpoint ────────────────────────────────────────

class TestSmokeRealCheckpoint:
    def test_load_and_eval(self):
        with open(MAIN_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        r = rows[0]  # first main run
        dn = r["dataset_name"]
        algo = r["algorithm"]
        cp = resolve_path(r["checkpoint_path"])
        ckpt = torch.load(cp, weights_only=False)
        model = MLP(OBS_DIM, N_ACTIONS, ckpt["config"]["hidden_dims"])
        model.load_state_dict(ckpt["model_state_dict"])
        sa, metrics = build_dataset_support(dn)
        mech = mechanism_eval(model, sa, algo, dn, metrics)
        assert 0 <= mech["ood_action_rate_step"] <= 1
        assert 0 <= mech["support_overlap_rate_unique_sa"] <= 1
        assert math.isfinite(mech["avg_return"])


# ── 9. Read-only constraint ──────────────────────────────────────────────────

class TestReadOnly:
    def test_no_writes(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_mechanism_analysis")
        src = inspect.getsource(mod)
        assert 'open(MANIFEST_PATH, "w"' not in src
        assert 'open(AUDIT_PATH, "w"' not in src
        assert 'open(MAIN_SUMMARY, "w"' not in src
        assert 'open(QUALITY_SUMMARY, "w"' not in src


# ── 10. No old-chain imports ─────────────────────────────────────────────────

class TestNoOldChain:
    def test_no_old_imports(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_envA_v2_mechanism_analysis")
        src = inspect.getsource(mod)
        assert "EnvA_main_rebuild" not in src
        assert "train_behavior_pool" not in src
        assert "benchmark" not in src.lower()
