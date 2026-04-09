"""
tests/test_envB_v2_dataset_prep.py
Dataset preparation tests for EnvB_v2 (frozen v4 config).

Checks:
  1. Wide dataset covers all 3 route families
  2. Narrow-A dataset covers only family A
  3. SA coverage gap is in the right direction (wide > narrow)
  4. Coverage values are in expected ball-park (not exact match to audit)
  5. .npz files can be round-tripped (saved and loaded correctly)
  6. No cross-contamination with final_datasets/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from scripts.build_envB_v2_datasets import (
    generate_dataset, compute_coverage, load_dataset, save_dataset,
    WIDE_CONFIG, NARROW_CONFIG, OUT_DIR, _ALL_SA, _EXCLUSIVE_SA,
)
from envs.gridworld_envs import EnvB_v2, N_ACTIONS

# ── Fixtures — generate once, reuse in multiple tests ─────────────────────────

@pytest.fixture(scope="module")
def wide_data():
    cfg = WIDE_CONFIG
    trans, vis = generate_dataset(cfg["families"], cfg["delay_prob"],
                                  cfg["n_trans"], cfg["seed"])
    cov = compute_coverage(vis)
    return trans, vis, cov


@pytest.fixture(scope="module")
def narrow_data():
    cfg = NARROW_CONFIG
    trans, vis = generate_dataset(cfg["families"], cfg["delay_prob"],
                                  cfg["n_trans"], cfg["seed"])
    cov = compute_coverage(vis)
    return trans, vis, cov


# ── 1. Wide dataset family coverage ───────────────────────────────────────────

class TestWideCoverage:

    def test_wide_covers_all_three_families(self, wide_data):
        _, vis, cov = wide_data
        assert cov["route_family_coverage"] == 1.0, (
            f"Wide should cover 3/3 families, got {cov['families_covered']}"
        )

    def test_wide_families_are_ABC(self, wide_data):
        _, vis, cov = wide_data
        assert set(cov["families_covered"]) == {"A", "B", "C"}

    def test_wide_sa_coverage_in_range(self, wide_data):
        _, _, cov = wide_data
        sa = cov["norm_sa_coverage"]
        # Expected ~0.24 from audit; allow wide tolerance for test determinism
        assert 0.15 <= sa <= 0.40, (
            f"Wide SA coverage {sa:.4f} outside expected [0.15, 0.40]"
        )

    def test_wide_not_saturated(self, wide_data):
        _, _, cov = wide_data
        assert cov["norm_sa_coverage"] < 0.85

    def test_wide_transition_count(self, wide_data):
        trans, _, _ = wide_data
        assert len(trans) == WIDE_CONFIG["n_trans"]

    def test_wide_covers_each_family_exclusively(self, wide_data):
        _, vis, _ = wide_data
        for fam, excl in _EXCLUSIVE_SA.items():
            covered = len(vis & excl)
            assert covered > 0, f"Wide dataset covers 0 exclusive SA for family {fam}"


# ── 2. Narrow-A dataset family coverage ───────────────────────────────────────

class TestNarrowCoverage:

    def test_narrow_covers_only_family_A(self, narrow_data):
        _, _, cov = narrow_data
        assert cov["route_family_coverage"] <= 0.50, (
            f"Narrow should cover ≤ 1/3 families, got {cov['families_covered']}"
        )
        assert "A" in cov["families_covered"], "Narrow should cover family A"

    def test_narrow_does_not_cover_B_exclusively(self, narrow_data):
        _, vis, _ = narrow_data
        # Family B exclusive cells should not appear in narrow-A data
        b_excl = _EXCLUSIVE_SA["B"]
        assert len(vis & b_excl) == 0, (
            "Narrow-A data contaminated with family B exclusive SA pairs"
        )

    def test_narrow_does_not_cover_C_exclusively(self, narrow_data):
        _, vis, _ = narrow_data
        c_excl = _EXCLUSIVE_SA["C"]
        assert len(vis & c_excl) == 0, (
            "Narrow-A data contaminated with family C exclusive SA pairs"
        )

    def test_narrow_sa_coverage_below_ceiling(self, narrow_data):
        _, _, cov = narrow_data
        assert cov["norm_sa_coverage"] <= 0.20, (
            f"Narrow SA coverage {cov['norm_sa_coverage']:.4f} exceeds ceiling 0.20"
        )

    def test_narrow_transition_count(self, narrow_data):
        trans, _, _ = narrow_data
        assert len(trans) == NARROW_CONFIG["n_trans"]

    def test_narrow_sa_coverage_nonzero(self, narrow_data):
        _, _, cov = narrow_data
        assert cov["norm_sa_coverage"] > 0.0


# ── 3. Coverage gap direction ──────────────────────────────────────────────────

class TestCoverageGap:

    def test_wide_sa_greater_than_narrow_sa(self, wide_data, narrow_data):
        _, _, wcov = wide_data
        _, _, ncov = narrow_data
        assert wcov["norm_sa_coverage"] > ncov["norm_sa_coverage"], (
            f"Coverage direction wrong: wide={wcov['norm_sa_coverage']:.4f} "
            f"<= narrow={ncov['norm_sa_coverage']:.4f}"
        )

    def test_coverage_gap_above_minimum(self, wide_data, narrow_data):
        _, _, wcov = wide_data
        _, _, ncov = narrow_data
        gap = wcov["norm_sa_coverage"] - ncov["norm_sa_coverage"]
        assert gap >= 0.10, (
            f"Coverage gap {gap:.4f} below minimum threshold 0.10"
        )

    def test_coverage_gap_above_conservative_floor(self, wide_data, narrow_data):
        # Tests gap >= 0.12 as a conservative structural floor.
        # The submission-worthy threshold (0.15) is verified at the frozen seeds
        # (425/411) and documented in ENVB_V2_PRE_FORMAL_RECONCILIATION.md.
        # Unit tests use a seed-robust lower bound (0.12) to avoid fragility.
        _, _, wcov = wide_data
        _, _, ncov = narrow_data
        gap = wcov["norm_sa_coverage"] - ncov["norm_sa_coverage"]
        assert gap >= 0.12, (
            f"Coverage gap {gap:.4f} unexpectedly low (structural floor: 0.12)"
        )

    def test_coverage_ratio_above_2(self, wide_data, narrow_data):
        _, _, wcov = wide_data
        _, _, ncov = narrow_data
        ratio = wcov["norm_sa_coverage"] / ncov["norm_sa_coverage"]
        assert ratio >= 2.0, (
            f"Coverage ratio {ratio:.2f}x below minimum 2.0"
        )


# ── 4. Data format and integrity ──────────────────────────────────────────────

class TestDataFormat:

    def test_transitions_are_tuples_of_5(self, wide_data):
        trans, _, _ = wide_data
        for t in trans[:10]:
            assert len(t) == 5, "Transition must be (pos, action, reward, npos, terminal)"

    def test_actions_in_valid_range(self, wide_data):
        trans, _, _ = wide_data
        for pos, action, _, _, _ in trans[:100]:
            assert 0 <= action < N_ACTIONS

    def test_rewards_are_float(self, wide_data):
        trans, _, _ = wide_data
        for _, _, reward, _, _ in trans[:100]:
            assert isinstance(reward, float)

    def test_rewards_correct_values(self, wide_data):
        trans, _, _ = wide_data
        # Dataset generator uses: +1.0 at goal, -0.01 per step (matching env convention)
        for _, _, reward, npos, terminal in trans[:1000]:
            if terminal:
                # Goal reached: +1.0 reward (generator uses +1 directly; env class uses
                # GOAL_REWARD + STEP_PENALTY = 0.99, but generator has its own logic)
                assert reward == 1.0 or abs(reward - 0.99) < 1e-6, (
                    f"Unexpected terminal reward {reward}"
                )
            else:
                assert abs(reward - (-0.01)) < 1e-6, (
                    f"Unexpected step reward {reward} (terminal={terminal})"
                )

    def test_terminal_flags_binary(self, wide_data):
        trans, _, _ = wide_data
        for _, _, _, _, terminal in trans[:100]:
            assert terminal in (True, False, 0, 1)

    def test_positions_in_reachable_set(self, wide_data):
        trans, _, _ = wide_data
        env = EnvB_v2()
        reachable = env.get_reachable_states()
        for pos, _, _, npos, _ in trans[:200]:
            assert pos  in reachable, f"Obs {pos} not reachable"
            assert npos in reachable, f"Next obs {npos} not reachable"


# ── 5. NPZ round-trip ─────────────────────────────────────────────────────────

class TestNpzRoundTrip:

    def test_save_and_load_wide(self, wide_data, tmp_path):
        trans, vis, cov = wide_data
        # Temporarily change OUT_DIR via monkeypatching is complex; just test shapes
        import scripts.build_envB_v2_datasets as bds
        orig_out = bds.OUT_DIR
        bds.OUT_DIR = str(tmp_path)
        try:
            path = save_dataset(trans, "test_wide", cov)
            obs, acts, rews, nobs, terms = load_dataset("test_wide")
        finally:
            bds.OUT_DIR = orig_out
        assert obs.shape  == (len(trans), 2)
        assert acts.shape == (len(trans),)
        assert rews.shape == (len(trans),)
        assert nobs.shape == (len(trans), 2)
        assert terms.shape == (len(trans),)
        assert obs.dtype   == np.int32
        assert acts.dtype  == np.int64
        assert rews.dtype  == np.float32

    def test_save_and_load_narrow(self, narrow_data, tmp_path):
        trans, vis, cov = narrow_data
        import scripts.build_envB_v2_datasets as bds
        orig_out = bds.OUT_DIR
        bds.OUT_DIR = str(tmp_path)
        try:
            path = save_dataset(trans, "test_narrow", cov)
            obs, acts, rews, nobs, terms = load_dataset("test_narrow")
        finally:
            bds.OUT_DIR = orig_out
        assert obs.shape == (len(trans), 2)
        assert acts.shape == (len(trans),)


# ── 6. No contamination of final_datasets ─────────────────────────────────────

class TestIsolation:

    def test_out_dir_is_not_final_datasets(self):
        """EnvB_v2 datasets must go to envB_v2_datasets/, never to final_datasets/."""
        assert "final_datasets" not in OUT_DIR
        assert "envB_v2_datasets" in OUT_DIR

    def test_dataset_names_do_not_match_envA_pattern(self, wide_data, narrow_data):
        """EnvB_v2 dataset names must not collide with EnvA_v2 conventions."""
        assert WIDE_CONFIG["name"].startswith("envB_v2_")
        assert NARROW_CONFIG["name"].startswith("envB_v2_")
