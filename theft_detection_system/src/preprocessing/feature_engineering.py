"""
feature_engineering.py
=======================
Production-quality FeatureEngineer class for the SGCC Electricity Theft
Detection system.

Each row in the input matrix ``X`` represents a single consumer's daily
electricity consumption readings over a period (e.g., 1 035 days in the
original SGCC dataset).  The class computes a rich set of hand-crafted
features that complement learned representations.

Typical usage
-------------
>>> from theft_detection_system.src.preprocessing.feature_engineering import FeatureEngineer
>>> fe = FeatureEngineer(enable_fft=True, fft_top_k=10)
>>> X_train_feat = fe.fit_transform(X_train)
>>> X_val_feat   = fe.transform(X_val)
>>> X_test_feat  = fe.transform(X_test)
>>> print(fe.get_feature_names())
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_fmt = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler.setFormatter(_fmt)
if not logger.handlers:
    logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------
class FeatureEngineer:
    """
    Compute hand-crafted feature groups from raw daily consumption time-series.

    Each method accepts a 2-D array ``X`` of shape ``(n_samples, n_days)``
    where each row is one consumer's ordered daily readings.

    Parameters
    ----------
    rolling_windows : list of int
        Window sizes (in days) for rolling statistics.  Default ``[7, 14, 30]``.
    enable_fft : bool
        Whether to compute FFT-based spectral features.  Default ``True``.
    fft_top_k : int
        Number of dominant frequency bins to retain.  Default ``10``.
    n_seasons : int
        Number of equally-sized seasons to divide the series into.
        Must evenly divide ``n_days`` or the feature group will be skipped.
        Default ``4``.
    scale_output : bool
        Whether to standardise the full feature matrix inside
        :meth:`fit_transform`.  Default ``False``.
    random_state : int
        Seed for any stochastic operations.  Default ``42``.
    """

    def __init__(
        self,
        rolling_windows: Sequence[int] = (7, 14, 30),
        enable_fft: bool = True,
        fft_top_k: int = 10,
        n_seasons: int = 4,
        scale_output: bool = False,
        random_state: int = 42,
    ) -> None:
        self.rolling_windows: List[int] = list(rolling_windows)
        self.enable_fft: bool = enable_fft
        self.fft_top_k: int = fft_top_k
        self.n_seasons: int = n_seasons
        self.scale_output: bool = scale_output
        self.random_state: int = random_state

        # Internal state set during fit_transform
        self._feature_names: List[str] = []
        self._output_scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

        np.random.seed(random_state)
        logger.info(
            "FeatureEngineer created  (rolling_windows=%s, fft=%s, top_k=%d)",
            self.rolling_windows, self.enable_fft, self.fft_top_k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        """Ensure ``X`` is a 2-D float32 array, coercing if necessary."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(
                f"Expected 2-D array, got {X.ndim}-D array of shape {X.shape}."
            )
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return X

    @staticmethod
    def _safe_nanmean(X: np.ndarray, axis: int = 1) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(X, axis=axis)

    @staticmethod
    def _safe_nanstd(X: np.ndarray, axis: int = 1) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanstd(X, axis=axis, ddof=1)

    # ------------------------------------------------------------------
    # 1. Statistical features
    # ------------------------------------------------------------------

    def compute_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-row summary statistics.

        Features (8 per sample):
          mean, median, max, min, std, variance, skewness, kurtosis

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)
            Raw daily consumption matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, 8)
        """
        X = self._validate_input(X)
        logger.debug("Computing statistical features …")

        mean_     = self._safe_nanmean(X, axis=1)
        median_   = np.nanmedian(X, axis=1)
        max_      = np.nanmax(X, axis=1)
        min_      = np.nanmin(X, axis=1)
        std_      = self._safe_nanstd(X, axis=1)
        var_      = np.nanvar(X, axis=1, ddof=1)

        # scipy skew / kurtosis – nan_policy='omit' skips NaNs per row
        skew_     = stats.skew(X, axis=1, nan_policy="omit")
        kurt_     = stats.kurtosis(X, axis=1, nan_policy="omit")

        # Replace any resulting NaN/Inf with 0
        feats = np.column_stack([mean_, median_, max_, min_, std_, var_, skew_, kurt_])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Trend features
    # ------------------------------------------------------------------

    def compute_trend_features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-row trend-based features using linear regression.

        Features (4 per sample):
          * ``slope``           – OLS slope of consumption vs. time index
          * ``daily_change``    – mean day-over-day absolute change
          * ``peak_to_avg``     – max / mean (load factor indicator)
          * ``load_factor``     – mean / max  (inverse; 0 when max=0)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)

        Returns
        -------
        np.ndarray, shape (n_samples, 4)
        """
        X = self._validate_input(X)
        n_samples, n_days = X.shape
        logger.debug("Computing trend features …")

        t = np.arange(n_days, dtype=np.float32)  # time index

        slopes = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            row = X[i]
            valid_mask = ~np.isnan(row)
            if valid_mask.sum() >= 2:
                slope, _, _, _, _ = stats.linregress(t[valid_mask], row[valid_mask])
                slopes[i] = float(slope)

        # Daily change (mean |diff| across each row)
        diffs = np.abs(np.diff(np.nan_to_num(X, nan=0.0), axis=1))
        daily_change = diffs.mean(axis=1)

        mean_ = self._safe_nanmean(X, axis=1)
        max_  = np.nanmax(X, axis=1)

        # Peak-to-average ratio (guard division by zero)
        peak_to_avg = np.where(mean_ != 0, max_ / mean_, 0.0).astype(np.float32)
        load_factor = np.where(max_ != 0, mean_ / max_, 0.0).astype(np.float32)

        feats = np.column_stack([slopes, daily_change, peak_to_avg, load_factor])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Rolling features
    # ------------------------------------------------------------------

    def compute_rolling_features(
        self,
        X: np.ndarray,
        windows: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute rolling mean and rolling variance for each window size.

        For each window ``w`` and each row the rolling statistics are computed,
        then the *mean of all rolling means* and *mean of all rolling variances*
        over the entire series are returned as two scalar features.

        This yields ``2 × len(windows)`` features per sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)
        windows : list of int, optional
            Override :attr:`rolling_windows`.

        Returns
        -------
        np.ndarray, shape (n_samples, 2 * len(windows))
        """
        X = self._validate_input(X)
        if windows is None:
            windows = self.rolling_windows
        logger.debug("Computing rolling features for windows=%s …", windows)

        n_samples, n_days = X.shape
        feature_list: List[np.ndarray] = []

        for w in windows:
            if w > n_days:
                logger.warning(
                    "Window %d > n_days %d; skipping this window.", w, n_days
                )
                # Pad with zeros to keep shape consistent
                feature_list.append(np.zeros((n_samples, 1), dtype=np.float32))
                feature_list.append(np.zeros((n_samples, 1), dtype=np.float32))
                continue

            roll_means = np.zeros(n_samples, dtype=np.float32)
            roll_vars  = np.zeros(n_samples, dtype=np.float32)

            for i in range(n_samples):
                row = X[i]
                # Sliding window mean / variance across valid positions
                segments = np.lib.stride_tricks.sliding_window_view(row, w)
                seg_means = np.nanmean(segments, axis=1)
                seg_vars  = np.nanvar(segments, axis=1, ddof=1 if w > 1 else 0)
                roll_means[i] = float(np.nanmean(seg_means))
                roll_vars[i]  = float(np.nanmean(seg_vars))

            feature_list.append(roll_means.reshape(-1, 1))
            feature_list.append(roll_vars.reshape(-1, 1))

        feats = np.hstack(feature_list)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Weekly patterns
    # ------------------------------------------------------------------

    def compute_weekly_patterns(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate daily readings into 7-day weeks and compute weekly summary.

        For each week position (day-of-week 0–6) compute the mean consumption
        across all weeks, giving a profile of 7 values per consumer.

        Additionally returns:
          * Mean weekly total
          * Std of weekly totals
          * Max weekly total
          * Min weekly total

        Total features: ``7 (day-of-week profile) + 4 (weekly-total stats)``.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)

        Returns
        -------
        np.ndarray, shape (n_samples, 11)
        """
        X = self._validate_input(X)
        n_samples, n_days = X.shape
        logger.debug("Computing weekly pattern features …")

        # Truncate to full weeks
        n_full_weeks = n_days // 7
        if n_full_weeks == 0:
            logger.warning(
                "n_days=%d < 7; returning zero weekly features.", n_days
            )
            return np.zeros((n_samples, 11), dtype=np.float32)

        X_trunc = X[:, : n_full_weeks * 7]  # (n_samples, n_full_weeks * 7)

        # Reshape to (n_samples, n_full_weeks, 7)
        X_weeks = X_trunc.reshape(n_samples, n_full_weeks, 7)

        # Day-of-week profile: mean across weeks
        dow_profile = np.nanmean(X_weeks, axis=1)  # (n_samples, 7)

        # Weekly totals: (n_samples, n_full_weeks)
        weekly_totals = np.nansum(X_weeks, axis=2)
        wt_mean = np.nanmean(weekly_totals, axis=1).reshape(-1, 1)
        wt_std  = self._safe_nanstd(weekly_totals, axis=1).reshape(-1, 1)
        wt_max  = np.nanmax(weekly_totals, axis=1).reshape(-1, 1)
        wt_min  = np.nanmin(weekly_totals, axis=1).reshape(-1, 1)

        feats = np.hstack([dow_profile, wt_mean, wt_std, wt_max, wt_min])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Monthly patterns
    # ------------------------------------------------------------------

    def compute_monthly_patterns(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate daily readings into 30-day months and compute monthly stats.

        Features per sample:
          * Mean monthly consumption
          * Std of monthly consumption
          * Max monthly consumption
          * Min monthly consumption
          * Month-over-month growth (first → last month)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)

        Returns
        -------
        np.ndarray, shape (n_samples, 5)
        """
        X = self._validate_input(X)
        n_samples, n_days = X.shape
        logger.debug("Computing monthly pattern features …")

        month_size = 30
        n_months = n_days // month_size

        if n_months == 0:
            logger.warning(
                "n_days=%d < 30; returning zero monthly features.", n_days
            )
            return np.zeros((n_samples, 5), dtype=np.float32)

        X_trunc = X[:, : n_months * month_size]
        X_months = X_trunc.reshape(n_samples, n_months, month_size)

        monthly_totals = np.nansum(X_months, axis=2)  # (n_samples, n_months)

        m_mean   = np.nanmean(monthly_totals, axis=1).reshape(-1, 1)
        m_std    = self._safe_nanstd(monthly_totals, axis=1).reshape(-1, 1)
        m_max    = np.nanmax(monthly_totals, axis=1).reshape(-1, 1)
        m_min    = np.nanmin(monthly_totals, axis=1).reshape(-1, 1)

        first_month = monthly_totals[:, 0]
        last_month  = monthly_totals[:, -1]
        mom_growth = np.where(
            first_month != 0,
            (last_month - first_month) / (np.abs(first_month) + 1e-9),
            0.0,
        ).reshape(-1, 1).astype(np.float32)

        feats = np.hstack([m_mean, m_std, m_max, m_min, mom_growth])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Seasonal features
    # ------------------------------------------------------------------

    def compute_seasonal_features(self, X: np.ndarray) -> np.ndarray:
        """
        Divide the time series into ``n_seasons`` equal-length seasons and
        compute per-season mean consumption.

        Also returns the ratio of each season's mean to the annual mean
        (season anomaly index), yielding ``2 × n_seasons`` features total.

        If ``n_days`` is not evenly divisible by ``n_seasons``, the surplus
        tail days are discarded.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)

        Returns
        -------
        np.ndarray, shape (n_samples, 2 * n_seasons)
        """
        X = self._validate_input(X)
        n_samples, n_days = X.shape
        n_seasons = self.n_seasons
        logger.debug("Computing seasonal features (n_seasons=%d) …", n_seasons)

        season_len = n_days // n_seasons
        if season_len == 0:
            logger.warning(
                "n_days=%d < n_seasons=%d; returning zero seasonal features.",
                n_days, n_seasons,
            )
            return np.zeros((n_samples, 2 * n_seasons), dtype=np.float32)

        X_trunc = X[:, : n_seasons * season_len]
        X_seasons = X_trunc.reshape(n_samples, n_seasons, season_len)

        season_means = np.nanmean(X_seasons, axis=2)  # (n_samples, n_seasons)
        annual_mean  = np.nanmean(season_means, axis=1, keepdims=True)  # (n_samples, 1)

        anomaly_idx = np.where(
            annual_mean != 0,
            season_means / (annual_mean + 1e-9),
            0.0,
        ).astype(np.float32)

        feats = np.hstack([season_means, anomaly_idx])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # 7. Growth rate
    # ------------------------------------------------------------------

    def compute_growth_rate(self, X: np.ndarray) -> np.ndarray:
        """
        Compute percentage change from the first non-NaN value to the last
        non-NaN value in each row.

        Features (2 per sample):
          * ``total_growth_rate`` – (last − first) / |first|
          * ``signed_growth``     – raw (last − first)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
        """
        X = self._validate_input(X)
        n_samples, _ = X.shape
        logger.debug("Computing growth rate features …")

        total_growth = np.zeros(n_samples, dtype=np.float32)
        signed_growth = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            row = X[i]
            valid_indices = np.where(~np.isnan(row))[0]
            if len(valid_indices) >= 2:
                first_val = row[valid_indices[0]]
                last_val  = row[valid_indices[-1]]
                signed_growth[i] = float(last_val - first_val)
                if first_val != 0:
                    total_growth[i] = float(
                        (last_val - first_val) / (abs(first_val) + 1e-9)
                    )

        feats = np.column_stack([total_growth, signed_growth])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # 8. FFT / spectral features
    # ------------------------------------------------------------------

    def compute_fft_features(
        self,
        X: np.ndarray,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute FFT on each row and return the top-*k* dominant frequency
        magnitudes, their corresponding frequency bin indices (normalised),
        and the total spectral energy.

        Features per sample: ``2 × top_k + 1``
          * ``top_k`` frequency magnitudes (sorted descending)
          * ``top_k`` normalised frequency bin indices
          * Total spectral energy (sum of squared magnitudes)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)
        top_k : int, optional
            Number of dominant components.  Defaults to :attr:`fft_top_k`.

        Returns
        -------
        np.ndarray, shape (n_samples, 2 * top_k + 1)
        """
        X = self._validate_input(X)
        if top_k is None:
            top_k = self.fft_top_k

        n_samples, n_days = X.shape
        n_freq = n_days // 2  # One-sided spectrum length
        actual_k = min(top_k, n_freq)
        logger.debug("Computing FFT features (top_k=%d) …", actual_k)

        freq_bins = np.fft.fftfreq(n_days)[: n_freq]  # Normalised frequencies

        top_magnitudes  = np.zeros((n_samples, actual_k), dtype=np.float32)
        top_freq_indices = np.zeros((n_samples, actual_k), dtype=np.float32)
        spectral_energy = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            row = np.nan_to_num(X[i], nan=0.0)
            spectrum = np.abs(fft(row))[: n_freq]
            energy   = float(np.sum(spectrum ** 2))
            spectral_energy[i] = energy

            # Top-k by magnitude
            top_idx = np.argsort(spectrum)[::-1][: actual_k]
            top_idx_sorted = top_idx[np.argsort(top_idx)]  # keep positional order
            top_magnitudes[i] = spectrum[top_idx_sorted]
            top_freq_indices[i] = freq_bins[top_idx_sorted]

        feats = np.hstack([top_magnitudes, top_freq_indices, spectral_energy.reshape(-1, 1)])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Compute all enabled features on the training data.

        Optionally fit a :class:`~sklearn.preprocessing.StandardScaler`
        on the final feature matrix if :attr:`scale_output` is ``True``.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_days)

        Returns
        -------
        np.ndarray, shape (n_samples, n_total_features)
        """
        logger.info("fit_transform called on X_train shape=%s", X_train.shape)
        X_train = self._validate_input(X_train)

        parts, names = self._compute_all_features(X_train)
        feats = np.hstack(parts)

        self._feature_names = names
        logger.info(
            "Total engineered features: %d  (names built)", len(self._feature_names)
        )

        if self.scale_output:
            self._output_scaler = StandardScaler()
            feats = self._output_scaler.fit_transform(feats)
            logger.info("Output scaler fitted and applied.")

        self._is_fitted = True
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply feature engineering to validation / test data without re-fitting
        any internal scalers.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)

        Returns
        -------
        np.ndarray, shape (n_samples, n_total_features)

        Raises
        ------
        RuntimeError
            If :meth:`fit_transform` has not been called first.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEngineer has not been fitted.  "
                "Call fit_transform(X_train) before transform()."
            )
        logger.info("transform called on X shape=%s", X.shape)
        X = self._validate_input(X)

        parts, _ = self._compute_all_features(X)
        feats = np.hstack(parts)

        if self.scale_output and self._output_scaler is not None:
            feats = self._output_scaler.transform(feats)

        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    # _compute_all_features  (internal)
    # ------------------------------------------------------------------

    def _compute_all_features(
        self, X: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Run all feature-group computations and collect their outputs together
        with generated feature names.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_days)

        Returns
        -------
        parts : list of np.ndarray
        names : list of str
        """
        parts: List[np.ndarray] = []
        names: List[str] = []

        # 1. Statistical
        stat = self.compute_statistical_features(X)
        parts.append(stat)
        stat_names = ["mean", "median", "max", "min", "std", "variance", "skewness", "kurtosis"]
        names.extend([f"stat_{n}" for n in stat_names])
        logger.debug("Statistical features: %d", stat.shape[1])

        # 2. Trend
        trend = self.compute_trend_features(X)
        parts.append(trend)
        trend_names = ["slope", "daily_change", "peak_to_avg", "load_factor"]
        names.extend([f"trend_{n}" for n in trend_names])
        logger.debug("Trend features: %d", trend.shape[1])

        # 3. Rolling
        rolling = self.compute_rolling_features(X)
        parts.append(rolling)
        for w in self.rolling_windows:
            names.append(f"rolling_mean_w{w}")
            names.append(f"rolling_var_w{w}")
        logger.debug("Rolling features: %d", rolling.shape[1])

        # 4. Weekly
        weekly = self.compute_weekly_patterns(X)
        parts.append(weekly)
        weekly_names = [f"dow_mean_day{d}" for d in range(7)] + [
            "weekly_total_mean", "weekly_total_std", "weekly_total_max", "weekly_total_min"
        ]
        names.extend([f"weekly_{n}" for n in weekly_names])
        logger.debug("Weekly features: %d", weekly.shape[1])

        # 5. Monthly
        monthly = self.compute_monthly_patterns(X)
        parts.append(monthly)
        monthly_names = ["monthly_mean", "monthly_std", "monthly_max", "monthly_min", "mom_growth"]
        names.extend([f"monthly_{n}" for n in monthly_names])
        logger.debug("Monthly features: %d", monthly.shape[1])

        # 6. Seasonal
        seasonal = self.compute_seasonal_features(X)
        parts.append(seasonal)
        for s in range(self.n_seasons):
            names.append(f"seasonal_mean_s{s}")
        for s in range(self.n_seasons):
            names.append(f"seasonal_anomaly_s{s}")
        logger.debug("Seasonal features: %d", seasonal.shape[1])

        # 7. Growth rate
        growth = self.compute_growth_rate(X)
        parts.append(growth)
        names.extend(["growth_total_growth_rate", "growth_signed_growth"])
        logger.debug("Growth rate features: %d", growth.shape[1])

        # 8. FFT (optional)
        if self.enable_fft:
            fft_feats = self.compute_fft_features(X, top_k=self.fft_top_k)
            parts.append(fft_feats)
            actual_k = min(self.fft_top_k, X.shape[1] // 2)
            for k in range(actual_k):
                names.append(f"fft_mag_{k}")
            for k in range(actual_k):
                names.append(f"fft_freq_{k}")
            names.append("fft_spectral_energy")
            logger.debug("FFT features: %d", fft_feats.shape[1])

        return parts, names

    # ------------------------------------------------------------------
    # get_feature_names
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """
        Return the ordered list of all engineered feature names.

        Returns
        -------
        list of str

        Raises
        ------
        RuntimeError
            If :meth:`fit_transform` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Feature names are only available after calling fit_transform()."
            )
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> Optional[int]:
        """Number of engineered features (``None`` before fitting)."""
        return len(self._feature_names) if self._is_fitted else None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FeatureEngineer("
            f"rolling_windows={self.rolling_windows}, "
            f"enable_fft={self.enable_fft}, "
            f"fft_top_k={self.fft_top_k}, "
            f"n_seasons={self.n_seasons}, "
            f"fitted={self._is_fitted})"
        )
