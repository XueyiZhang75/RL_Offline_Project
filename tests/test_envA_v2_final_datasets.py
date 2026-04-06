"""
tests/test_envA_v2_final_datasets.py
Clean Phase 5: verify dataset generation frozen rules and structure.
"""

import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from scripts.generate_envA_v2_final_datasets import (
    GEN_SEEDS, DATASET_SPECS,
    NARROW_MEDIUM_IDS, WIDE_MEDIUM_IDS,
    WIDE_SUBOPTIMAL_IDS, WIDE_EXPERT_IDS,
    MIXED_TIER_CYCLE,
    CATALOG_PATH, PILOT_SUMMARY, MANIFEST_PATH,
    TOTAL_SA, TOTAL_STATES,
    load_catalog_selected, generate_dataset,
)
from scripts.verify_envA_v2_proxy_gate import (
    FAMILIES as P2_FAMILIES,
    SEED_FAMILY_MAP as P2_MAP,
)
from scripts.generate_envA_v2_final_datasets import (
    FAMILIES as GEN_FAMILIES,
    SEED_FAMILY_MAP as GEN_MAP,
)


# ── 1. Frozen semantics inherited ────────────────────────────────────────────

class TestFrozenSemantics:
    def test_families_is_phase2(self):
        assert GEN_FAMILIES is P2_FAMILIES

    def test_seed_map_is_phase2(self):
        assert GEN_MAP is P2_MAP


# ── 2. Source compositions frozen ─────────────────────────────────────────────

class TestSourceCompositions:
    def test_narrow_medium(self):
        assert NARROW_MEDIUM_IDS == ["envA_v2_seed0_medium_controller"]

    def test_wide_medium_6(self):
        assert len(WIDE_MEDIUM_IDS) == 6
        assert WIDE_MEDIUM_IDS[0] == "envA_v2_seed0_medium_controller"
        assert WIDE_MEDIUM_IDS[-1] == "envA_v2_seed6_medium_controller"

    def test_wide_suboptimal_6(self):
        assert len(WIDE_SUBOPTIMAL_IDS) == 6

    def test_wide_expert_6(self):
        assert len(WIDE_EXPERT_IDS) == 6

    def test_mixed_tier_cycle(self):
        assert MIXED_TIER_CYCLE == ["random", "medium", "medium", "expert"]


# ── 3. Generation seeds frozen ───────────────────────────────────────────────

class TestGenSeeds:
    def test_9_seeds(self):
        assert len(GEN_SEEDS) == 9

    def test_specific_seeds(self):
        assert GEN_SEEDS["envA_v2_small_wide_medium"] == 100
        assert GEN_SEEDS["envA_v2_quality_mixed_wide50k"] == 114

    def test_all_unique(self):
        assert len(set(GEN_SEEDS.values())) == 9


# ── 4. Dataset specs frozen ──────────────────────────────────────────────────

class TestDatasetSpecs:
    def test_9_specs(self):
        assert len(DATASET_SPECS) == 9

    def test_main_four_targets(self):
        names = {s["name"]: s["target"] for s in DATASET_SPECS}
        assert names["envA_v2_small_wide_medium"] == 50_000
        assert names["envA_v2_large_wide_medium"] == 200_000
        assert names["envA_v2_small_narrow_medium"] == 50_000
        assert names["envA_v2_large_narrow_medium"] == 200_000

    def test_quality_all_50k(self):
        for s in DATASET_SPECS:
            if "quality" in s["name"]:
                assert s["target"] == 50_000


# ── 5. Pre-flight conditions ─────────────────────────────────────────────────

class TestPreFlight:
    def test_catalog_has_24_selected(self):
        cat = load_catalog_selected("EnvA_v2")
        assert len(cat) == 24

    def test_catalog_bins_8_each(self):
        cat = load_catalog_selected("EnvA_v2")
        for qbin in ["suboptimal", "medium", "expert"]:
            cnt = sum(1 for r in cat.values() if r["quality_bin"] == qbin)
            assert cnt == 8, f"{qbin} count {cnt}"

    def test_pilot_passed(self):
        with open(PILOT_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        agg = [r for r in rows if r["record_type"] == "aggregate"]
        assert len(agg) == 1
        assert agg[0]["all_3_pass"] == "yes"


# ── 6. Rollout produces correct structure ─────────────────────────────────────

class TestRolloutStructure:
    @pytest.fixture(scope="class")
    def small_data(self):
        cat = load_catalog_selected("EnvA_v2")
        from scripts.generate_envA_v2_final_datasets import load_artifact, art_to_ctrl
        ctrl = art_to_ctrl(load_artifact(cat[NARROW_MEDIUM_IDS[0]]))
        return generate_dataset(lambda ep: ctrl, 500, 999)

    def test_11_keys(self, small_data):
        required = {"observations", "actions", "rewards", "next_observations",
                     "terminals", "truncations", "episode_ids", "timesteps",
                     "source_policy_ids", "source_train_seeds",
                     "source_behavior_epsilons"}
        assert required.issubset(set(small_data.keys()))

    def test_obs_shape(self, small_data):
        assert small_data["observations"].ndim == 2
        assert small_data["observations"].shape[1] == 2

    def test_dtypes(self, small_data):
        assert small_data["observations"].dtype == np.int32
        assert small_data["actions"].dtype == np.int32
        assert small_data["rewards"].dtype == np.float32
        assert small_data["terminals"].dtype == bool

    def test_actual_ge_target(self, small_data):
        assert small_data["actual_transitions"] >= 500

    def test_coverage_in_range(self, small_data):
        assert 0 < small_data["norm_sa_cov"] < 1


# ── 7. No old-chain imports ──────────────────────────────────────────────────

class TestNoOldChain:
    def test_no_dqn(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.generate_envA_v2_final_datasets")
        src = inspect.getsource(mod)
        assert "train_behavior_pool" not in src
        assert "DQN" not in src
        assert "EnvA_main_rebuild" not in src

    def test_no_old_generate(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.generate_envA_v2_final_datasets")
        src = inspect.getsource(mod)
        assert "generate_final_datasets" not in src.replace(
            "generate_envA_v2_final_datasets", "")
