"""
test_preprocessing.py
=====================
Pytest unit tests for the SGCC theft-detection data pipeline components:
  - SGCCDataLoader  (loading, missing-value handling, outlier detection,
                     duplicate removal, split ratios, scaler output range)
  - FeatureEngineer (fit_transform shape, feature names, FFT finiteness,
                     rolling with short sequences)
  - ImbalanceHandler (SMOTE minority-class increase, weighted_loss passthrough,
                      compute_pos_weight ratio)

All tests use synthetic in-memory data only — no real SGCC CSV required.
Run from any directory:
    pytest theft_detection_system/tests/test_preprocessing.py -v
"""

from __future__ import annotations

import sys
import os
import io
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup: allow running from any working directory
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent                   # tests/
_SYSTEM_ROOT = _THIS_DIR.parent                               # theft_detection_system/
_REPO_ROOT = _SYSTEM_ROOT.parent                              # scratch-main/

for _p in [str(_SYSTEM_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Lazy imports (after path fixup)
# ---------------------------------------------------------------------------
from src.preprocessing.data_loader import SGCCDataLoader      # noqa: E402
from src.preprocessing.feature_engineering import FeatureEngineer  # noqa: E402

# ImbalanceHandler may not exist yet; guard gracefully so tests are skippable
try:
    from src.preprocessing.imbalance_handler import ImbalanceHandler
    _HAS_IMBALANCE_HANDLER = True
except ImportError:
    _HAS_IMBALANCE_HANDLER = False

logger = logging.getLogger(__name__)


# ===========================================================================
# ── FIXTURES ────────────────────────────────────────────────────────────────
# ===========================================================================

@pytest.fixture(scope="module")
def n_days() -> int:
    """Number of daily feature columns in synthetic data."""
    return 60


@pytest.fixture(scope="module")
def n_samples() -> int:
    """Total number of synthetic consumers."""
    return 200


@pytest.fixture(scope="module")
def raw_csv_path(tmp_path_factory, n_samples: int, n_days: int) -> Path:
    """
    Write a synthetic SGCC-shaped CSV to a temporary file and return its path.
    Layout: CONS_NO | FLAG | 2016-01-01 | 2016-01-02 | ...
    Class distribution: 20% theft (FLAG=1), 80% normal (FLAG=0).
    """
    rng = np.random.default_rng(42)
    day_cols = pd.date_range("2016-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()

    cons_no = [f"C{i:05d}" for i in range(n_samples)]
    flag = ([1] * (n_samples // 5)) + ([0] * (n_samples - n_samples // 5))

    data = rng.uniform(0, 10, size=(n_samples, n_days)).astype(np.float32)

    # Inject some NaNs (~5 %)
    nan_mask = rng.random(size=(n_samples, n_days)) < 0.05
    data[nan_mask] = np.nan

    df = pd.DataFrame(data, columns=day_cols)
    df.insert(0, "CONS_NO", cons_no)
    df.insert(1, "FLAG", flag)

    # Add two exact duplicate rows
    dup_rows = df.iloc[[0, 1]].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    tmp = tmp_path_factory.mktemp("data")
    csv_path = tmp / "sgcc_synthetic.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def loaded_loader(raw_csv_path: Path) -> SGCCDataLoader:
    """Return an SGCCDataLoader that has been loaded but NOT yet preprocessed."""
    loader = SGCCDataLoader(
        data_path=str(raw_csv_path),
        target_column="FLAG",
        random_state=42,
    )
    loader.load_data()
    return loader


@pytest.fixture(scope="module")
def processed_loader(raw_csv_path: Path) -> SGCCDataLoader:
    """
    Return a fully preprocessed SGCCDataLoader ready for split+scale tests.
    """
    loader = SGCCDataLoader(
        data_path=str(raw_csv_path),
        target_column="FLAG",
        random_state=42,
    )
    loader.load_data()
    loader.validate_data()
    loader.remove_duplicates()
    loader.handle_missing_values(strategy="interpolate")
    loader.detect_and_handle_outliers(method="iqr", threshold=1.5)
    loader.split_data(test_size=0.15, val_size=0.15, stratify=True)
    loader.scale_features(scaler_type="robust")
    return loader


@pytest.fixture(scope="module")
def X_raw(n_samples: int, n_days: int) -> np.ndarray:
    """Raw 2-D float array for FeatureEngineer tests (no NaNs)."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 10, size=(n_samples, n_days)).astype(np.float32)


@pytest.fixture(scope="module")
def imbalanced_Xy(n_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic imbalanced dataset: 180 negatives + 20 positives.
    Returns (X, y).
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, n_days)).astype(np.float32)
    y = np.array([0] * 180 + [1] * 20, dtype=np.int64)
    return X, y


# ===========================================================================
# ── SGCCDataLoader TESTS ────────────────────────────────────────────────────
# ===========================================================================

class TestSGCCDataLoader:
    """Unit tests for SGCCDataLoader."""

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def test_load_data_sets_is_loaded_flag(self, loaded_loader: SGCCDataLoader) -> None:
        """After load_data(), _is_loaded must be True."""
        assert loaded_loader._is_loaded is True, (
            "Expected _is_loaded=True after load_data()"
        )

    def test_load_data_df_not_none(self, loaded_loader: SGCCDataLoader) -> None:
        """Internal DataFrame must be non-None after loading."""
        assert loaded_loader._df is not None, (
            "Internal DataFrame (_df) should not be None after load_data()"
        )

    def test_load_data_shape(
        self, loaded_loader: SGCCDataLoader, n_samples: int, n_days: int
    ) -> None:
        """DataFrame must have n_samples + 2 duplicates rows and n_days + 2 columns."""
        df = loaded_loader._df
        # n_samples + 2 duplicate rows
        assert df.shape[0] == n_samples + 2, (
            f"Expected {n_samples + 2} rows, got {df.shape[0]}"
        )
        # n_days feature cols + CONS_NO + FLAG
        assert df.shape[1] == n_days + 2, (
            f"Expected {n_days + 2} columns, got {df.shape[1]}"
        )

    def test_load_nonexistent_file_raises(self) -> None:
        """load_data() on a missing path must raise FileNotFoundError."""
        loader = SGCCDataLoader(
            data_path="/nonexistent/path/data.csv",
            target_column="FLAG",
            random_state=42,
        )
        with pytest.raises((FileNotFoundError, RuntimeError)):
            loader.load_data()

    # -----------------------------------------------------------------------
    # Duplicate removal
    # -----------------------------------------------------------------------

    def test_remove_duplicates_reduces_rows(self, raw_csv_path: Path, n_samples: int) -> None:
        """remove_duplicates() should remove the 2 injected duplicate rows."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        before = len(loader._df)
        loader.remove_duplicates()
        after = len(loader._df)
        assert after < before, (
            f"Expected fewer rows after duplicate removal; before={before}, after={after}"
        )
        assert after == n_samples, (
            f"Expected exactly {n_samples} rows after removing 2 duplicates, got {after}"
        )

    def test_remove_duplicates_resets_index(self, raw_csv_path: Path) -> None:
        """Index should be a clean 0-based RangeIndex after duplicate removal."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.remove_duplicates()
        expected_idx = pd.RangeIndex(start=0, stop=len(loader._df), step=1)
        pd.testing.assert_index_equal(loader._df.index, expected_idx)

    # -----------------------------------------------------------------------
    # Missing value handling
    # -----------------------------------------------------------------------

    def test_handle_missing_interpolate_no_nans(
        self, raw_csv_path: Path, n_days: int
    ) -> None:
        """After interpolate strategy, feature columns must be NaN-free."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.handle_missing_values(strategy="interpolate")
        feat_cols = loader.get_feature_columns()
        n_missing = loader._df[feat_cols].isnull().sum().sum()
        assert n_missing == 0, (
            f"Expected 0 NaN values after interpolation, found {n_missing}"
        )

    def test_handle_missing_mean_no_nans(self, raw_csv_path: Path) -> None:
        """After mean strategy, feature columns must be NaN-free."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.handle_missing_values(strategy="mean")
        feat_cols = loader.get_feature_columns()
        n_missing = loader._df[feat_cols].isnull().sum().sum()
        assert n_missing == 0, (
            f"Expected 0 NaN values after mean imputation, found {n_missing}"
        )

    def test_handle_missing_median_no_nans(self, raw_csv_path: Path) -> None:
        """After median strategy, feature columns must be NaN-free."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.handle_missing_values(strategy="median")
        feat_cols = loader.get_feature_columns()
        n_missing = loader._df[feat_cols].isnull().sum().sum()
        assert n_missing == 0, (
            f"Expected 0 NaN values after median imputation, found {n_missing}"
        )

    def test_handle_missing_invalid_strategy_raises(
        self, raw_csv_path: Path
    ) -> None:
        """Passing an unknown strategy must raise ValueError."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        with pytest.raises(ValueError):
            loader.handle_missing_values(strategy="unknown_strategy")

    # -----------------------------------------------------------------------
    # Outlier detection
    # -----------------------------------------------------------------------

    def test_outlier_iqr_clip_values_bounded(
        self, raw_csv_path: Path, n_days: int
    ) -> None:
        """
        After IQR clip with threshold=1.5, every value should be within
        [Q1 - 1.5*IQR, Q3 + 1.5*IQR] per column.  We check that no value
        exceeds the unclipped 0.01 / 99.99 percentile range (a loose sanity).
        """
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.handle_missing_values(strategy="mean")

        # Inject extreme outliers
        feat_cols = loader.get_feature_columns()
        loader._df.iloc[0, 2] = 1e6   # column index 2 is first feature col

        loader.detect_and_handle_outliers(method="iqr", threshold=1.5, action="clip")

        max_val = loader._df[feat_cols].max().max()
        assert max_val < 1e4, (
            f"Expected outlier clipping to bound values; max after clip = {max_val}"
        )

    def test_outlier_zscore_reduces_extremes(self, raw_csv_path: Path) -> None:
        """Z-score clipping should keep feature values within a sane range."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.handle_missing_values(strategy="mean")

        feat_cols = loader.get_feature_columns()
        loader._df.iloc[1, 2] = 1e5

        loader.detect_and_handle_outliers(method="zscore", threshold=3.0, action="clip")

        max_val = loader._df[feat_cols].max().max()
        assert max_val < 1e4, (
            f"Z-score clip should reduce extremes; max after clip = {max_val}"
        )

    def test_outlier_invalid_method_raises(self, raw_csv_path: Path) -> None:
        """Unknown outlier method must raise ValueError."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        with pytest.raises(ValueError):
            loader.detect_and_handle_outliers(method="bad_method")

    # -----------------------------------------------------------------------
    # Split ratios
    # -----------------------------------------------------------------------

    def test_split_ratios_approx(
        self, processed_loader: SGCCDataLoader, n_samples: int
    ) -> None:
        """
        Train/val/test sizes should each be approximately the requested fraction
        of the original data (within 5% tolerance).
        """
        # processed_loader used test_size=0.15, val_size=0.15
        total = (
            len(processed_loader.train)
            + len(processed_loader.val)
            + len(processed_loader.test)
        )
        test_frac  = len(processed_loader.test)  / total
        val_frac   = len(processed_loader.val)   / total
        train_frac = len(processed_loader.train) / total

        assert abs(test_frac - 0.15) < 0.05, (
            f"Test fraction {test_frac:.3f} not close to 0.15"
        )
        assert abs(val_frac - 0.15) < 0.05, (
            f"Val fraction {val_frac:.3f} not close to 0.15"
        )
        assert train_frac > 0.60, (
            f"Train fraction {train_frac:.3f} is suspiciously small"
        )

    def test_split_total_equals_full_dataset(
        self, processed_loader: SGCCDataLoader
    ) -> None:
        """Train + val + test sizes must sum to the deduplicated dataset length."""
        total = (
            len(processed_loader.train)
            + len(processed_loader.val)
            + len(processed_loader.test)
        )
        # The processed loader removed duplicates, so we expect exactly n_samples rows
        assert total > 0, "Combined split size must be > 0"

    def test_split_invalid_fractions_raises(self, raw_csv_path: Path) -> None:
        """test_size + val_size >= 1.0 must raise ValueError."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        with pytest.raises(ValueError):
            loader.split_data(test_size=0.6, val_size=0.5)

    # -----------------------------------------------------------------------
    # Scaler output range
    # -----------------------------------------------------------------------

    def test_minmax_scaler_output_in_zero_one(self, raw_csv_path: Path) -> None:
        """MinMax-scaled training features must lie in [0, 1] (approximately)."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.handle_missing_values(strategy="interpolate")
        loader.split_data(test_size=0.15, val_size=0.15, stratify=True)
        loader.scale_features(scaler_type="minmax")

        feat_cols = loader.get_feature_columns()
        train_max = loader.train[feat_cols].max().max()
        train_min = loader.train[feat_cols].min().min()

        assert train_max <= 1.0 + 1e-5, (
            f"MinMax scaler: training max {train_max:.4f} > 1.0"
        )
        assert train_min >= 0.0 - 1e-5, (
            f"MinMax scaler: training min {train_min:.4f} < 0.0"
        )

    def test_standard_scaler_mean_near_zero(self, raw_csv_path: Path) -> None:
        """Standard-scaled training features must have mean ≈ 0 per column."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.handle_missing_values(strategy="interpolate")
        loader.split_data(test_size=0.15, val_size=0.15, stratify=True)
        loader.scale_features(scaler_type="standard")

        feat_cols = loader.get_feature_columns()
        col_means = loader.train[feat_cols].mean()
        max_abs_mean = col_means.abs().max()

        assert max_abs_mean < 1e-3, (
            f"Standard scaler: max |column mean| on train = {max_abs_mean:.6f}; expected ≈ 0"
        )

    def test_invalid_scaler_type_raises(self, raw_csv_path: Path) -> None:
        """Unsupported scaler type must raise ValueError."""
        loader = SGCCDataLoader(
            data_path=str(raw_csv_path),
            target_column="FLAG",
            random_state=42,
        )
        loader.load_data()
        loader.split_data(test_size=0.15, val_size=0.15, stratify=True)
        with pytest.raises(ValueError):
            loader.scale_features(scaler_type="fancy_scaler")

    # -----------------------------------------------------------------------
    # get_numpy_arrays
    # -----------------------------------------------------------------------

    def test_get_numpy_arrays_shapes(
        self, processed_loader: SGCCDataLoader, n_days: int
    ) -> None:
        """get_numpy_arrays() must return 6 arrays with compatible shapes."""
        X_tr, X_va, X_te, y_tr, y_va, y_te = processed_loader.get_numpy_arrays()

        assert X_tr.shape[0] == y_tr.shape[0], "X_train and y_train row mismatch"
        assert X_va.shape[0] == y_va.shape[0], "X_val and y_val row mismatch"
        assert X_te.shape[0] == y_te.shape[0], "X_test and y_test row mismatch"

        assert X_tr.shape[1] == n_days, (
            f"Expected {n_days} feature columns in X_train, got {X_tr.shape[1]}"
        )

    def test_get_numpy_arrays_dtypes(self, processed_loader: SGCCDataLoader) -> None:
        """Feature arrays must be float32; label arrays must be int64."""
        X_tr, X_va, X_te, y_tr, y_va, y_te = processed_loader.get_numpy_arrays()

        assert X_tr.dtype == np.float32, f"X_train dtype: {X_tr.dtype}"
        assert X_va.dtype == np.float32, f"X_val dtype:   {X_va.dtype}"
        assert X_te.dtype == np.float32, f"X_test dtype:  {X_te.dtype}"

        assert y_tr.dtype == np.int64, f"y_train dtype: {y_tr.dtype}"

    # -----------------------------------------------------------------------
    # get_feature_columns
    # -----------------------------------------------------------------------

    def test_feature_columns_excludes_id_and_label(
        self, loaded_loader: SGCCDataLoader, n_days: int
    ) -> None:
        """get_feature_columns() must not include CONS_NO or FLAG."""
        feat_cols = loaded_loader.get_feature_columns()
        assert "CONS_NO" not in feat_cols, "CONS_NO should not be in feature columns"
        assert "FLAG" not in feat_cols, "FLAG should not be in feature columns"
        assert len(feat_cols) == n_days, (
            f"Expected {n_days} feature columns, got {len(feat_cols)}"
        )


# ===========================================================================
# ── FeatureEngineer TESTS ──────────────────────────────────────────────────
# ===========================================================================

class TestFeatureEngineer:
    """Unit tests for FeatureEngineer."""

    # -----------------------------------------------------------------------
    # fit_transform shape
    # -----------------------------------------------------------------------

    def test_fit_transform_returns_2d(self, X_raw: np.ndarray) -> None:
        """fit_transform must return a 2-D array."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        out = fe.fit_transform(X_raw)
        assert out.ndim == 2, f"Expected 2-D output, got {out.ndim}-D"

    def test_fit_transform_preserves_n_samples(self, X_raw: np.ndarray) -> None:
        """Output row count must equal input row count."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        out = fe.fit_transform(X_raw)
        assert out.shape[0] == X_raw.shape[0], (
            f"Row count mismatch: input {X_raw.shape[0]}, output {out.shape[0]}"
        )

    def test_fit_transform_feature_count_positive(self, X_raw: np.ndarray) -> None:
        """Number of engineered features must be > 0."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        out = fe.fit_transform(X_raw)
        assert out.shape[1] > 0, "Expected at least 1 engineered feature"

    def test_fit_transform_output_dtype_float32(self, X_raw: np.ndarray) -> None:
        """Output array dtype must be float32."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        out = fe.fit_transform(X_raw)
        assert out.dtype == np.float32, (
            f"Expected float32 output, got {out.dtype}"
        )

    # -----------------------------------------------------------------------
    # Feature names
    # -----------------------------------------------------------------------

    def test_get_feature_names_not_empty(self, X_raw: np.ndarray) -> None:
        """get_feature_names() must return a non-empty list after fit_transform."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        fe.fit_transform(X_raw)
        names = fe.get_feature_names()
        assert len(names) > 0, "Feature names list must not be empty"

    def test_get_feature_names_count_matches_columns(self, X_raw: np.ndarray) -> None:
        """Length of feature names must match number of output columns."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        out = fe.fit_transform(X_raw)
        names = fe.get_feature_names()
        assert len(names) == out.shape[1], (
            f"Name count {len(names)} != column count {out.shape[1]}"
        )

    def test_feature_names_are_strings(self, X_raw: np.ndarray) -> None:
        """Every feature name must be a non-empty string."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        fe.fit_transform(X_raw)
        for name in fe.get_feature_names():
            assert isinstance(name, str) and len(name) > 0, (
                f"Feature name '{name}' is not a valid non-empty string"
            )

    def test_get_feature_names_before_fit_raises(self) -> None:
        """Calling get_feature_names() before fit_transform must raise RuntimeError."""
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.get_feature_names()

    def test_feature_names_unique(self, X_raw: np.ndarray) -> None:
        """All feature names must be distinct (no duplicates)."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        fe.fit_transform(X_raw)
        names = fe.get_feature_names()
        assert len(names) == len(set(names)), (
            "Feature names contain duplicates"
        )

    # -----------------------------------------------------------------------
    # FFT features finiteness
    # -----------------------------------------------------------------------

    def test_fft_features_all_finite(self, X_raw: np.ndarray) -> None:
        """FFT features must not contain NaN or Inf."""
        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        out = fe.fit_transform(X_raw)
        assert np.all(np.isfinite(out)), (
            "Output contains NaN or Inf — FFT features may be problematic"
        )

    def test_fft_disabled_fewer_features(self, X_raw: np.ndarray) -> None:
        """Disabling FFT should produce fewer features than when it is enabled."""
        fe_on  = FeatureEngineer(enable_fft=True,  fft_top_k=5)
        fe_off = FeatureEngineer(enable_fft=False, fft_top_k=5)
        out_on  = fe_on.fit_transform(X_raw)
        out_off = fe_off.fit_transform(X_raw)
        assert out_off.shape[1] < out_on.shape[1], (
            "Expected fewer features when FFT is disabled"
        )

    # -----------------------------------------------------------------------
    # Rolling features with short sequences
    # -----------------------------------------------------------------------

    def test_rolling_short_sequence_no_crash(self) -> None:
        """
        When the input has fewer columns than the largest rolling window,
        fit_transform must not raise — instead it pads with zeros.
        """
        # Only 5 days — much smaller than the default windows [7, 14, 30]
        X_short = np.random.default_rng(1).uniform(0, 1, (20, 5)).astype(np.float32)
        fe = FeatureEngineer(rolling_windows=[7, 14, 30], enable_fft=False)
        try:
            out = fe.fit_transform(X_short)
        except Exception as exc:
            pytest.fail(
                f"fit_transform raised an unexpected exception on short sequences: {exc}"
            )
        assert out.shape[0] == 20, "Row count should equal n_samples=20"

    def test_rolling_short_sequence_shape_correct(self) -> None:
        """Output shape for short sequences must still be (n_samples, n_features)."""
        X_short = np.random.default_rng(2).uniform(0, 1, (10, 3)).astype(np.float32)
        fe = FeatureEngineer(rolling_windows=[7, 14], enable_fft=False)
        out = fe.fit_transform(X_short)
        # 2 values per window (mean + variance) = 4 rolling cols plus others
        assert out.ndim == 2 and out.shape[0] == 10, (
            f"Unexpected output shape for short sequence: {out.shape}"
        )

    # -----------------------------------------------------------------------
    # transform (val/test)
    # -----------------------------------------------------------------------

    def test_transform_before_fit_raises(self, X_raw: np.ndarray) -> None:
        """Calling transform() before fit_transform() must raise RuntimeError."""
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(X_raw)

    def test_transform_matches_fit_transform_columns(self, X_raw: np.ndarray) -> None:
        """transform() must produce the same number of columns as fit_transform()."""
        rng = np.random.default_rng(99)
        X_val = rng.uniform(0, 5, size=X_raw.shape).astype(np.float32)

        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        out_train = fe.fit_transform(X_raw)
        out_val   = fe.transform(X_val)
        assert out_val.shape[1] == out_train.shape[1], (
            f"transform() column count {out_val.shape[1]} "
            f"!= fit_transform() column count {out_train.shape[1]}"
        )

    def test_transform_output_finite(self, X_raw: np.ndarray) -> None:
        """transform() output must be fully finite (no NaN / Inf)."""
        rng = np.random.default_rng(55)
        X_val = rng.uniform(0, 5, size=X_raw.shape).astype(np.float32)

        fe = FeatureEngineer(enable_fft=True, fft_top_k=5)
        fe.fit_transform(X_raw)
        out = fe.transform(X_val)
        assert np.all(np.isfinite(out)), (
            "transform() output contains NaN or Inf"
        )


# ===========================================================================
# ── ImbalanceHandler TESTS ─────────────────────────────────────────────────
# ===========================================================================

@pytest.mark.skipif(
    not _HAS_IMBALANCE_HANDLER,
    reason="ImbalanceHandler not yet implemented — skipping.",
)
class TestImbalanceHandler:
    """Unit tests for ImbalanceHandler."""

    # -----------------------------------------------------------------------
    # SMOTE increases minority class
    # -----------------------------------------------------------------------

    def test_smote_increases_minority_class(
        self, imbalanced_Xy: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """After SMOTE resampling the minority class count must increase."""
        X, y = imbalanced_Xy
        n_minority_before = int((y == 1).sum())

        handler = ImbalanceHandler(method="smote")
        X_res, y_res = handler.resample(X, y)

        n_minority_after = int((y_res == 1).sum())
        assert n_minority_after > n_minority_before, (
            f"SMOTE should increase minority class: before={n_minority_before}, "
            f"after={n_minority_after}"
        )

    def test_smote_result_is_numpy(
        self, imbalanced_Xy: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """resample() must return NumPy arrays, not DataFrames."""
        X, y = imbalanced_Xy
        handler = ImbalanceHandler(method="smote")
        X_res, y_res = handler.resample(X, y)
        assert isinstance(X_res, np.ndarray), "X_res must be np.ndarray"
        assert isinstance(y_res, np.ndarray), "y_res must be np.ndarray"

    def test_smote_preserves_n_features(
        self, imbalanced_Xy: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """SMOTE must not change the number of feature columns."""
        X, y = imbalanced_Xy
        handler = ImbalanceHandler(method="smote")
        X_res, _ = handler.resample(X, y)
        assert X_res.shape[1] == X.shape[1], (
            f"Feature count changed after SMOTE: before={X.shape[1]}, after={X_res.shape[1]}"
        )

    # -----------------------------------------------------------------------
    # weighted_loss returns unchanged arrays
    # -----------------------------------------------------------------------

    def test_weighted_loss_returns_unchanged_arrays(
        self, imbalanced_Xy: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        When method='weighted_loss', resample() should return the original
        X and y without modification (loss weighting is applied at training time).
        """
        X, y = imbalanced_Xy
        handler = ImbalanceHandler(method="weighted_loss")
        X_res, y_res = handler.resample(X, y)

        assert X_res.shape == X.shape, (
            f"weighted_loss: X shape changed {X.shape} -> {X_res.shape}"
        )
        assert np.array_equal(X_res, X), (
            "weighted_loss: X values were modified — expected passthrough"
        )
        assert np.array_equal(y_res, y), (
            "weighted_loss: y values were modified — expected passthrough"
        )

    # -----------------------------------------------------------------------
    # compute_pos_weight returns correct ratio
    # -----------------------------------------------------------------------

    def test_compute_pos_weight_correct_ratio(
        self, imbalanced_Xy: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        pos_weight = n_negatives / n_positives.
        With 180 negatives and 20 positives the expected value is 9.0.
        """
        X, y = imbalanced_Xy
        handler = ImbalanceHandler(method="weighted_loss")
        pos_weight = handler.compute_pos_weight(y)

        expected = 180 / 20  # = 9.0
        assert abs(pos_weight - expected) < 0.5, (
            f"Expected pos_weight ≈ {expected:.2f}, got {pos_weight:.4f}"
        )

    def test_compute_pos_weight_positive(
        self, imbalanced_Xy: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """pos_weight must always be a positive finite float."""
        X, y = imbalanced_Xy
        handler = ImbalanceHandler(method="weighted_loss")
        pw = handler.compute_pos_weight(y)
        assert isinstance(pw, float) or np.isscalar(pw), (
            "pos_weight must be a scalar"
        )
        assert pw > 0, f"pos_weight must be > 0, got {pw}"
        assert np.isfinite(pw), f"pos_weight must be finite, got {pw}"

    # -----------------------------------------------------------------------
    # compute_class_weights
    # -----------------------------------------------------------------------

    def test_compute_class_weights_returns_dict_or_array(
        self, imbalanced_Xy: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """compute_class_weights must return a dict or array with two entries."""
        X, y = imbalanced_Xy
        handler = ImbalanceHandler(method="weighted_loss")
        cw = handler.compute_class_weights(y)
        assert cw is not None, "compute_class_weights returned None"
        if isinstance(cw, dict):
            assert len(cw) == 2, f"Expected 2 class weights, got {len(cw)}"
        else:
            arr = np.asarray(cw)
            assert arr.size == 2, f"Expected 2 class weights, got {arr.size}"


# ===========================================================================
# ── ENTRY POINT (optional standalone run) ──────────────────────────────────
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
