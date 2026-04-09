"""
tests/test_envC_v2_dataset_prep.py
Dataset preparation tests for EnvC_v2.

Checks:
  1. Wide dataset covers all 4 combined families (RFC = 1.0)
  2. Narrow-LU dataset covers only LU family (RFC ≤ 0.35)
  3. Wide SA coverage in gate range [0.18, 0.40]
  4. Narrow SA coverage below ceiling (≤ 0.20)
  5. Coverage gap ≥ 0.10
  6. Coverage ratio (wide/narrow) ≥ 2.0
  7. Wide covers both has_key=0 and has_key=1 phases (both > 0)
  8. Narrow does NOT contaminate Post-D exclusive SA pairs
  9. LU-narrow does NOT contaminate Pre-R or Post-D exclusive SA
 10. NPZ round-trip preserves shapes and dtypes
 11. Datasets are written to envC_v2_datasets/, not final_datasets/
 12. Coverage gap conservative floor (sanity bound)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from scripts.build_envC_v2_datasets import (
    build_all, generate_dataset, compute_coverage, load_dataset,
    WIDE_CONFIG, NARROW_CONFIG,
    _ALL_EXTENDED_SA, _EXCLUSIVE_SA_C2, _REACHABLE_C2,
    OUT_DIR,
)

# ── Fixtures: generate datasets once per session ──────────────────────────────

@pytest.fixture(scope="session")
def wide_data():
    trans, vis_sa = generate_dataset(
        WIDE_CONFIG["families"], WIDE_CONFIG["delay_prob"],
        WIDE_CONFIG["delay_type"], WIDE_CONFIG["n_trans"], WIDE_CONFIG["seed"])
    return trans, vis_sa, compute_coverage(vis_sa)


@pytest.fixture(scope="session")
def narrow_data():
    trans, vis_sa = generate_dataset(
        NARROW_CONFIG["families"], NARROW_CONFIG["delay_prob"],
        NARROW_CONFIG["delay_type"], NARROW_CONFIG["n_trans"], NARROW_CONFIG["seed"])
    return trans, vis_sa, compute_coverage(vis_sa)


# ── 1. Wide dataset covers all 4 families ─────────────────────────────────────

class TestWideCoverage:

    def test_wide_rfc_1(self, wide_data):
        _, _, cov = wide_data
        assert cov["route_family_coverage"] == 1.0, \
            f"Wide RFC should be 1.0, got {cov['route_family_coverage']}"

    def test_wide_covers_all_four_families(self, wide_data):
        _, _, cov = wide_data
        assert set(cov["families_covered"]) == {"LU", "LD", "RU", "RD"}

    def test_wide_sa_in_gate_range(self, wide_data):
        _, _, cov = wide_data
        sa = cov["norm_sa_coverage"]
        assert 0.18 <= sa <= 0.40, \
            f"Wide SA {sa:.4f} outside gate range [0.18, 0.40]"

    def test_wide_sa_above_minimum(self, wide_data):
        _, _, cov = wide_data
        assert cov["norm_sa_coverage"] >= 0.18

    def test_wide_not_saturated(self, wide_data):
        _, _, cov = wide_data
        assert cov["norm_sa_coverage"] <= 0.85

    def test_wide_hk0_coverage_positive(self, wide_data):
        _, _, cov = wide_data
        assert cov["cov_hk0"] > 0, "Wide must cover some has_key=0 SA pairs"

    def test_wide_hk1_coverage_positive(self, wide_data):
        _, _, cov = wide_data
        assert cov["cov_hk1"] > 0, "Wide must cover some has_key=1 SA pairs"

    def test_wide_covers_pre_l_exclusive_sa(self, wide_data):
        _, vis_sa, _ = wide_data
        pre_l = _EXCLUSIVE_SA_C2["Pre-L"]
        assert vis_sa & pre_l, "Wide must cover some Pre-L exclusive SA"

    def test_wide_covers_pre_r_exclusive_sa(self, wide_data):
        _, vis_sa, _ = wide_data
        pre_r = _EXCLUSIVE_SA_C2["Pre-R"]
        assert vis_sa & pre_r, "Wide must cover some Pre-R exclusive SA"

    def test_wide_covers_post_u_exclusive_sa(self, wide_data):
        _, vis_sa, _ = wide_data
        post_u = _EXCLUSIVE_SA_C2["Post-U"]
        assert vis_sa & post_u, "Wide must cover some Post-U exclusive SA"

    def test_wide_covers_post_d_exclusive_sa(self, wide_data):
        _, vis_sa, _ = wide_data
        post_d = _EXCLUSIVE_SA_C2["Post-D"]
        assert vis_sa & post_d, "Wide must cover some Post-D exclusive SA"


# ── 2. Narrow-LU dataset ──────────────────────────────────────────────────────

class TestNarrowCoverage:

    def test_narrow_rfc_lte_035(self, narrow_data):
        _, _, cov = narrow_data
        assert cov["route_family_coverage"] <= 0.35, \
            f"Narrow RFC {cov['route_family_coverage']:.3f} exceeds 0.35"

    def test_narrow_covers_only_lu(self, narrow_data):
        _, _, cov = narrow_data
        assert cov["families_covered"] == ["LU"]

    def test_narrow_sa_below_ceiling(self, narrow_data):
        _, _, cov = narrow_data
        assert cov["norm_sa_coverage"] <= 0.20, \
            f"Narrow SA {cov['norm_sa_coverage']:.4f} exceeds ceiling 0.20"

    def test_narrow_no_pre_r_contamination(self, narrow_data):
        """LU narrow should not visit Pre-R exclusive SA (those belong to R families)."""
        _, vis_sa, _ = narrow_data
        pre_r = _EXCLUSIVE_SA_C2["Pre-R"]
        contaminated = vis_sa & pre_r
        assert len(contaminated) == 0, \
            f"Narrow-LU contaminated {len(contaminated)} Pre-R SA pairs"

    def test_narrow_no_post_d_contamination(self, narrow_data):
        """LU narrow should not visit Post-D exclusive SA (those belong to D families)."""
        _, vis_sa, _ = narrow_data
        post_d = _EXCLUSIVE_SA_C2["Post-D"]
        contaminated = vis_sa & post_d
        assert len(contaminated) == 0, \
            f"Narrow-LU contaminated {len(contaminated)} Post-D SA pairs"

    def test_narrow_covers_post_u_exclusive_sa(self, narrow_data):
        """LU narrow SHOULD visit Post-U (LU is a Post-U family)."""
        _, vis_sa, _ = narrow_data
        post_u = _EXCLUSIVE_SA_C2["Post-U"]
        assert vis_sa & post_u, "Narrow-LU should cover some Post-U exclusive SA"

    def test_narrow_hk0_coverage_positive(self, narrow_data):
        _, _, cov = narrow_data
        assert cov["cov_hk0"] > 0

    def test_narrow_hk1_coverage_positive(self, narrow_data):
        _, _, cov = narrow_data
        assert cov["cov_hk1"] > 0


# ── 3. Gap / ratio ────────────────────────────────────────────────────────────

class TestCoverageContrast:

    def test_wide_greater_than_narrow(self, wide_data, narrow_data):
        _, _, wcov = wide_data; _, _, ncov = narrow_data
        assert wcov["norm_sa_coverage"] > ncov["norm_sa_coverage"]

    def test_coverage_gap_meets_gate(self, wide_data, narrow_data):
        _, _, wcov = wide_data; _, _, ncov = narrow_data
        gap = wcov["norm_sa_coverage"] - ncov["norm_sa_coverage"]
        assert gap >= 0.10, f"Coverage gap {gap:.4f} below gate threshold 0.10"

    def test_coverage_ratio_meets_gate(self, wide_data, narrow_data):
        _, _, wcov = wide_data; _, _, ncov = narrow_data
        ratio = wcov["norm_sa_coverage"] / ncov["norm_sa_coverage"]
        assert ratio >= 2.0, f"Coverage ratio {ratio:.2f}x below gate threshold 2.0"

    def test_coverage_gap_submission_worthy(self, wide_data, narrow_data):
        _, _, wcov = wide_data; _, _, ncov = narrow_data
        gap = wcov["norm_sa_coverage"] - ncov["norm_sa_coverage"]
        # Pilot confirmed submission-worthy (gap≥0.15)
        assert gap >= 0.15, \
            f"Coverage gap {gap:.4f} below submission-worthy threshold 0.15"

    def test_wide_hk1_greater_than_narrow_hk1(self, wide_data, narrow_data):
        _, _, wcov = wide_data; _, _, ncov = narrow_data
        assert wcov["cov_hk1"] > ncov["cov_hk1"]

    def test_wide_hk0_greater_than_narrow_hk0(self, wide_data, narrow_data):
        _, _, wcov = wide_data; _, _, ncov = narrow_data
        assert wcov["cov_hk0"] > ncov["cov_hk0"]


# ── 4. Data format ────────────────────────────────────────────────────────────

class TestDataFormat:

    def test_wide_transition_count(self, wide_data):
        trans, _, _ = wide_data
        assert len(trans) == WIDE_CONFIG["n_trans"]

    def test_narrow_transition_count(self, narrow_data):
        trans, _, _ = narrow_data
        assert len(trans) == NARROW_CONFIG["n_trans"]

    def test_wide_transition_tuple_format(self, wide_data):
        trans, _, _ = wide_data
        for state, action, reward, nstate, terminal in trans[:10]:
            pos, hk = state
            r, c = pos
            assert 0 < r < 20 and 0 < c < 20
            assert hk in (0, 1)
            assert 0 <= action < 4
            assert reward in (-0.01, 1.0) or abs(reward + 0.01) < 1e-5

    def test_actions_in_range(self, wide_data):
        trans, _, _ = wide_data
        actions = [t[1] for t in trans]
        assert min(actions) >= 0
        assert max(actions) <= 3

    def test_terminals_binary(self, wide_data):
        trans, _, _ = wide_data
        terminals = [t[4] for t in trans]
        for t in terminals:
            assert t in (True, False, 0, 1)


# ── 5. NPZ round-trip ─────────────────────────────────────────────────────────

class TestNpzRoundTrip:

    @pytest.fixture(scope="class")
    def built_datasets(self):
        build_all(verbose=False)

    def test_wide_npz_exists(self, built_datasets):
        path = os.path.join(OUT_DIR, "envC_v2_small_wide.npz")
        assert os.path.exists(path)

    def test_narrow_npz_exists(self, built_datasets):
        path = os.path.join(OUT_DIR, "envC_v2_large_narrow_LU.npz")
        assert os.path.exists(path)

    def test_manifest_exists(self, built_datasets):
        path = os.path.join(OUT_DIR, "envC_v2_dataset_manifest.csv")
        assert os.path.exists(path)

    def test_wide_load_shapes(self, built_datasets):
        obs, acts, rews, nobs, terms = load_dataset("envC_v2_small_wide")
        assert obs.shape == (WIDE_CONFIG["n_trans"], 3)
        assert acts.shape == (WIDE_CONFIG["n_trans"],)
        assert nobs.shape == (WIDE_CONFIG["n_trans"], 3)

    def test_wide_obs_has_key_column(self, built_datasets):
        obs, *_ = load_dataset("envC_v2_small_wide")
        # Third column is has_key: must be 0 or 1
        hk_col = obs[:, 2]
        assert np.all((hk_col == 0) | (hk_col == 1)), \
            "has_key column must be 0 or 1"

    def test_narrow_load_shapes(self, built_datasets):
        obs, acts, rews, nobs, terms = load_dataset("envC_v2_large_narrow_LU")
        assert obs.shape == (NARROW_CONFIG["n_trans"], 3)

    def test_wide_dtypes(self, built_datasets):
        obs, acts, rews, nobs, terms = load_dataset("envC_v2_small_wide")
        assert obs.dtype == np.int32
        assert acts.dtype == np.int64
        assert rews.dtype == np.float32
        assert terms.dtype == np.float32


# ── 6. Isolation from other datasets ─────────────────────────────────────────

class TestIsolation:

    def test_output_dir_is_envc_v2_datasets(self):
        assert "envC_v2_datasets" in OUT_DIR

    def test_output_dir_not_final_datasets(self):
        assert "final_datasets" not in OUT_DIR

    def test_no_envb_v2_contamination(self):
        # EnvB_v2 dataset dir should not be affected
        envb_dir = OUT_DIR.replace("envC_v2_datasets", "envB_v2_datasets")
        if os.path.exists(envb_dir):
            files = os.listdir(envb_dir)
            # EnvC_v2 datasets should not appear in EnvB_v2 dir
            for f in files:
                assert "envC_v2" not in f, \
                    f"EnvC_v2 file found in EnvB_v2 dir: {f}"
