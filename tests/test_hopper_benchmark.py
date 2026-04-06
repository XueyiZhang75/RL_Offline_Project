"""
tests/test_hopper_benchmark.py
Clean Phase 12 重执行版: verify frozen benchmark scope, resumable execution,
append-per-run durability, scratch cleanup, path normalization, and minimal runnability.
"""

import sys, os, csv, math, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from scripts.run_hopper_benchmark import (
    BENCHMARK_SEEDS, BENCHMARK_DATASETS, ALGORITHMS,
    BENCHMARK_EVAL_EPISODES, ALGO_N_STEPS, TOTAL_RUNS,
    SEED_COLUMNS, SUMMARY_COLUMNS,
    AUDIT_PATH, MAIN_SUMMARY, VAL_SUMMARY, QUALITY_SUMMARY, MECHANISM_SUMMARY,
    BC_CFG, CQL_CFG, IQL_CFG, TD3BC_CFG,
    normalize_score, HOPPER_RANDOM_SCORE, HOPPER_EXPERT_SCORE,
    download_dataset, load_d4rl_dataset, create_algo, evaluate_policy,
    run_is_valid, load_completed_runs, append_seed_row,
    cleanup_backend_scratch, normalize_ckpt_path, resolve_path,
    existing_benchmark_complete,
)


# ── 1. Frozen constants ──────────────────────────────────────────────────────

class TestFrozenConstants:
    def test_seeds_0_to_4(self):
        assert BENCHMARK_SEEDS == list(range(5))

    def test_3_hopper_datasets(self):
        assert len(BENCHMARK_DATASETS) == 3
        assert "hopper-medium"        in BENCHMARK_DATASETS
        assert "hopper-medium-replay" in BENCHMARK_DATASETS
        assert "hopper-medium-expert" in BENCHMARK_DATASETS

    def test_no_walker(self):
        for ds in BENCHMARK_DATASETS:
            assert "walker" not in ds.lower()

    def test_no_halfcheetah(self):
        for ds in BENCHMARK_DATASETS:
            assert "cheetah" not in ds.lower()

    def test_4_algorithms(self):
        assert ALGORITHMS == ["bc", "cql", "iql", "td3bc"]

    def test_eval_episodes_20(self):
        assert BENCHMARK_EVAL_EPISODES == 20

    def test_n_steps_100k(self):
        assert ALGO_N_STEPS == 100_000

    def test_total_runs_60(self):
        assert TOTAL_RUNS == 60  # 3 datasets x 4 algos x 5 seeds

    def test_4_algo_configs_are_dicts_with_batch_size(self):
        for name, cfg in [("BC", BC_CFG), ("CQL", CQL_CFG), ("IQL", IQL_CFG), ("TD3BC", TD3BC_CFG)]:
            assert isinstance(cfg, dict), f"{name}_CFG is not a dict"
            assert "batch_size" in cfg,   f"{name}_CFG missing batch_size"

    def test_algo_configs_independent(self):
        # Each config is a separate object
        cfgs = [BC_CFG, CQL_CFG, IQL_CFG, TD3BC_CFG]
        for i, a in enumerate(cfgs):
            for j, b in enumerate(cfgs):
                if i != j:
                    assert a is not b


# ── 2. CSV schemas ───────────────────────────────────────────────────────────

class TestSchemas:
    def test_seed_columns_required_fields(self):
        for col in ["normalized_score", "raw_return", "checkpoint_path", "status",
                    "dataset_name", "algorithm", "train_seed"]:
            assert col in SEED_COLUMNS

    def test_summary_columns_required_fields(self):
        for col in ["mean_normalized_score", "ci95_normalized_score_low", "n_runs",
                    "dataset_name", "algorithm"]:
            assert col in SUMMARY_COLUMNS

    def test_summary_gate_is_12_rows(self):
        # 3 datasets x 4 algorithms = 12
        assert len(BENCHMARK_DATASETS) * len(ALGORITHMS) == 12


# ── 3. Normalization ─────────────────────────────────────────────────────────

class TestNormalization:
    def test_random_score_zero(self):
        assert normalize_score(HOPPER_RANDOM_SCORE) == pytest.approx(0.0)

    def test_expert_score_100(self):
        assert normalize_score(HOPPER_EXPERT_SCORE) == pytest.approx(100.0)

    def test_midpoint(self):
        mid = (HOPPER_RANDOM_SCORE + HOPPER_EXPERT_SCORE) / 2
        assert normalize_score(mid) == pytest.approx(50.0)


# ── 4. Pre-flight conditions ─────────────────────────────────────────────────

class TestPreFlight:
    def test_audit_frozen(self):
        with open(AUDIT_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 13
        assert all(r["freeze_ready"] == "yes" for r in rows)

    def test_main_completed(self):
        with open(MAIN_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        assert all(r["status"] == "completed" for r in rows)

    def test_validation_completed(self):
        with open(VAL_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 160
        assert all(r["status"] == "completed" for r in rows)

    def test_quality_completed(self):
        with open(QUALITY_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 200
        assert all(r["status"] == "completed" for r in rows)

    def test_mechanism_exists(self):
        with open(MECHANISM_SUMMARY, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 18


# ── 5. Resume logic ──────────────────────────────────────────────────────────

class TestResumeLogic:
    def _row(self, ckpt="", status="completed", score="42.5"):
        r = {c: "" for c in SEED_COLUMNS}
        r.update({"status": status, "checkpoint_path": ckpt, "normalized_score": score})
        return r

    def test_empty_ckpt_path_invalid(self):
        assert not run_is_valid(self._row(ckpt=""))

    def test_missing_ckpt_invalid(self):
        assert not run_is_valid(self._row(ckpt="nonexistent/path.pt"))

    def test_failed_status_invalid(self):
        assert not run_is_valid(self._row(ckpt="nonexistent", status="failed"))

    def test_nan_score_invalid(self):
        assert not run_is_valid(self._row(ckpt="nonexistent", score="nan"))

    def test_inf_score_invalid(self):
        assert not run_is_valid(self._row(ckpt="nonexistent", score="inf"))

    def test_valid_run_with_existing_dir(self, tmp_path):
        # d3rlpy saves checkpoints as directories
        ckpt = str(tmp_path / "fake.d3")
        os.makedirs(ckpt)
        assert run_is_valid(self._row(ckpt=ckpt, score="42.5"))

    def test_valid_run_with_existing_file(self, tmp_path):
        ckpt = str(tmp_path / "fake.pt")
        ckpt_path = ckpt.replace("\\", "/")
        with open(ckpt, "w") as f:
            f.write("x")
        assert run_is_valid(self._row(ckpt=ckpt_path, score="10.0"))


# ── 6. Append-after-each-run durability ──────────────────────────────────────

class TestAppendAfterRun:
    def test_creates_file_with_header(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "seed_results.csv")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV", csv_path)
        row = {c: f"v{i}" for i, c in enumerate(SEED_COLUMNS)}
        append_seed_row(row)
        assert os.path.isfile(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row

    def test_appends_multiple_rows_correctly(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "seed_results.csv")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV", csv_path)
        for i in range(5):
            row = {c: str(i) for c in SEED_COLUMNS}
            append_seed_row(row)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_header_appears_only_once(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "seed_results.csv")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV", csv_path)
        for i in range(3):
            row = {c: str(i) for c in SEED_COLUMNS}
            append_seed_row(row)
        with open(csv_path) as f:
            content = f.read()
        assert content.count(SEED_COLUMNS[0]) == 1

    def test_file_is_readable_after_append(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "seed_results.csv")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV", csv_path)
        row = {c: "test_val" for c in SEED_COLUMNS}
        append_seed_row(row)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0][SEED_COLUMNS[0]] == "test_val"


# ── 7. Scratch log cleanup ───────────────────────────────────────────────────

class TestScratchCleanup:
    def test_removes_scratch_dir(self, tmp_path, monkeypatch):
        scratch = str(tmp_path / "_backend_tmp")
        os.makedirs(scratch)
        monkeypatch.setattr("scripts.run_hopper_benchmark.SCRATCH_DIR", scratch)
        cleanup_backend_scratch()
        assert not os.path.isdir(scratch)

    def test_no_error_if_scratch_absent(self, tmp_path, monkeypatch):
        scratch = str(tmp_path / "does_not_exist")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SCRATCH_DIR", scratch)
        cleanup_backend_scratch()  # must not raise

    def test_removes_nested_scratch_content(self, tmp_path, monkeypatch):
        scratch = str(tmp_path / "_backend_tmp")
        os.makedirs(os.path.join(scratch, "run_001"))
        (open(os.path.join(scratch, "log.txt"), "w")).close()
        monkeypatch.setattr("scripts.run_hopper_benchmark.SCRATCH_DIR", scratch)
        cleanup_backend_scratch()
        assert not os.path.isdir(scratch)


# ── 8. Path normalization ────────────────────────────────────────────────────

class TestPathNormalization:
    def test_normalize_ckpt_path_no_backslash(self):
        import scripts.run_hopper_benchmark as mod
        p = os.path.join(mod.BENCH_DIR, "test.d3")
        rel = normalize_ckpt_path(p)
        assert "\\" not in rel

    def test_normalize_ckpt_path_is_relative(self):
        import scripts.run_hopper_benchmark as mod
        p = os.path.join(mod.BENCH_DIR, "test.d3")
        rel = normalize_ckpt_path(p)
        assert not os.path.isabs(rel)

    def test_resolve_path_backslash(self):
        p = resolve_path("artifacts\\training_benchmark\\test.d3")
        assert "training_benchmark" in p

    def test_resolve_path_forward_slash(self):
        p = resolve_path("artifacts/training_benchmark/test.d3")
        assert "training_benchmark" in p

    def test_resolve_abs_path_unchanged_modulo_normpath(self, tmp_path):
        abs_p = str(tmp_path / "foo.pt").replace("\\", "/")
        resolved = resolve_path(abs_p)
        assert os.path.isabs(resolved)


# ── 9. Load completed runs ───────────────────────────────────────────────────

class TestLoadCompletedRuns:
    def test_empty_dict_if_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV",
                            str(tmp_path / "nonexistent.csv"))
        assert load_completed_runs() == {}

    def test_loads_keyed_correctly(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "test.csv")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV", csv_path)
        row = {c: "" for c in SEED_COLUMNS}
        row.update({"dataset_name": "hopper-medium", "algorithm": "bc",
                    "train_seed": "0", "status": "completed"})
        append_seed_row(row)
        completed = load_completed_runs()
        assert ("hopper-medium", "bc", "0") in completed

    def test_last_row_wins_on_duplicate_key(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "test.csv")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV", csv_path)
        for status in ["failed", "completed"]:
            row = {c: "" for c in SEED_COLUMNS}
            row.update({"dataset_name": "hopper-medium", "algorithm": "bc",
                        "train_seed": "0", "status": status})
            append_seed_row(row)
        completed = load_completed_runs()
        assert completed[("hopper-medium", "bc", "0")]["status"] == "completed"


# ── 10. Reuse / verify mode ──────────────────────────────────────────────────

class TestReuseMode:
    def test_incomplete_not_detected(self, tmp_path, monkeypatch):
        fake_csv = str(tmp_path / "fake_seed.csv")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV", fake_csv)
        ok, _ = existing_benchmark_complete()
        assert not ok

    def test_partial_not_detected(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "partial.csv")
        monkeypatch.setattr("scripts.run_hopper_benchmark.SEED_CSV", csv_path)
        # Write only 3 rows (not 60)
        for i in range(3):
            row = {c: str(i) for c in SEED_COLUMNS}
            row["status"] = "completed"
            append_seed_row(row)
        ok, rows = existing_benchmark_complete()
        assert not ok


# ── 11. Backend availability ─────────────────────────────────────────────────

class TestBackend:
    def test_d3rlpy_importable(self):
        import d3rlpy
        assert hasattr(d3rlpy, "__version__")

    def test_noop_adapter_factory_available(self):
        from d3rlpy.logging import NoopAdapterFactory
        assert NoopAdapterFactory is not None

    def test_dataset_downloadable(self):
        path = download_dataset("hopper-medium")
        assert os.path.exists(path)

    def test_dataset_loadable(self):
        path = download_dataset("hopper-medium")
        ds = load_d4rl_dataset(path)
        assert ds.transition_count > 0


# ── 12. Algorithm creation ────────────────────────────────────────────────────

class TestAlgoCreation:
    def test_bc_creates(self):
        assert create_algo("bc", 0) is not None

    def test_cql_creates(self):
        assert create_algo("cql", 0) is not None

    def test_iql_creates(self):
        assert create_algo("iql", 0) is not None

    def test_td3bc_creates(self):
        assert create_algo("td3bc", 0) is not None

    def test_unknown_algo_raises(self):
        with pytest.raises(ValueError):
            create_algo("ppo", 0)


# ── 13. Smoke test ───────────────────────────────────────────────────────────

class TestSmoke:
    def test_bc_mini_train_and_eval(self):
        import h5py, d3rlpy
        from d3rlpy.logging import NoopAdapterFactory
        path = download_dataset("hopper-medium")
        with h5py.File(path, "r") as f:
            tiny = d3rlpy.dataset.MDPDataset(
                observations=np.array(f["observations"][:500], dtype=np.float32),
                actions=np.array(f["actions"][:500],      dtype=np.float32),
                rewards=np.array(f["rewards"][:500],      dtype=np.float32),
                terminals=np.array([i == 499 for i in range(500)], dtype=bool),
            )
        algo = create_algo("bc", 0)
        algo.fit(tiny, n_steps=10, show_progress=False,
                 logger_adapter=NoopAdapterFactory())
        rets = evaluate_policy(algo, "Hopper-v4", 1)
        assert len(rets) == 1
        assert math.isfinite(rets[0])
        assert math.isfinite(normalize_score(rets[0]))


# ── 14. No old-chain imports ─────────────────────────────────────────────────

class TestNoOldChain:
    def test_no_old_imports(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_hopper_benchmark")
        src = inspect.getsource(mod)
        assert "EnvA_main_rebuild"  not in src
        assert "train_behavior_pool" not in src
        assert "walker"             not in src.lower()
        assert "halfcheetah"        not in src.lower()

    def test_no_audit_write(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_hopper_benchmark")
        src = inspect.getsource(mod)
        assert 'open(AUDIT_PATH, "w"' not in src

    def test_no_manifest_reference(self):
        import importlib, inspect
        mod = importlib.import_module("scripts.run_hopper_benchmark")
        src = inspect.getsource(mod)
        assert "MANIFEST_PATH" not in src
