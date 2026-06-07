"""
data_loader.py
==============
Production-quality DataLoader class for the SGCC Electricity Theft Detection dataset.
Handles loading, validation, cleaning, splitting, scaling, and serialisation of the
raw SGCC CSV data.

Typical usage
-------------
>>> from theft_detection_system.src.preprocessing.data_loader import DataLoader
>>> dl = DataLoader(random_state=42)
>>> dl.load_data("path/to/sgcc.csv")
>>> dl.validate_data()
>>> dl.remove_duplicates()
>>> dl.handle_missing_values(strategy="interpolate")
>>> dl.detect_and_handle_outliers(method="iqr", threshold=1.5)
>>> dl.split_data(test_size=0.15, val_size=0.15, stratify=True)
>>> dl.scale_features(scaler_type="robust")
>>> X_train, X_val, X_test, y_train, y_val, y_test = dl.get_numpy_arrays()
>>> summary = dl.get_data_summary()
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
_console_handler.setFormatter(_fmt)
if not logger.handlers:
    logger.addHandler(_console_handler)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class DataLoader:
    """
    End-to-end data loading and preprocessing pipeline for the SGCC
    electricity theft detection dataset.

    The SGCC dataset ships as a wide CSV where:
      * Column ``CONS_NO``  – consumer ID
      * Column ``FLAG``     – label (0 = normal, 1 = theft)
      * All other columns  – daily electricity consumption readings

    Parameters
    ----------
    random_state : int, optional
        Seed for all random operations.  Default ``42``.
    """

    # Column name constants (case-insensitive search is applied at load time)
    _ID_COL: str = "CONS_NO"
    _LABEL_COL: str = "FLAG"

    # Supported scalers
    _SCALER_MAP: Dict[str, type] = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }

    def __init__(self, random_state: int = 42) -> None:
        self.random_state: int = random_state
        np.random.seed(random_state)

        # Raw data
        self._df: Optional[pd.DataFrame] = None

        # Detected column names (may differ in case)
        self._id_col: Optional[str] = None
        self._label_col: Optional[str] = None

        # Split DataFrames
        self._train: Optional[pd.DataFrame] = None
        self._val: Optional[pd.DataFrame] = None
        self._test: Optional[pd.DataFrame] = None

        # Fitted scaler
        self._scaler: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]] = None
        self._scaler_type: Optional[str] = None

        # Flags
        self._is_loaded: bool = False
        self._is_split: bool = False
        self._is_scaled: bool = False

        logger.info("DataLoader initialised with random_state=%d", random_state)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        """Raise RuntimeError if data has not been loaded yet."""
        if not self._is_loaded or self._df is None:
            raise RuntimeError(
                "Data not loaded.  Call load_data(path) first."
            )

    def _assert_split(self) -> None:
        """Raise RuntimeError if data has not been split yet."""
        if not self._is_split:
            raise RuntimeError(
                "Data not split.  Call split_data() first."
            )

    # ------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------

    def load_data(self, path: Union[str, Path]) -> "DataLoader":
        """
        Load SGCC data from a CSV file.

        Automatically detects the consumer-ID column (``CONS_NO``) and the
        label column (``FLAG``) using a case-insensitive search.

        Parameters
        ----------
        path : str or Path
            Path to the CSV file.

        Returns
        -------
        self
            Returns the instance so calls can be chained.

        Raises
        ------
        FileNotFoundError
            If *path* does not point to an existing file.
        ValueError
            If the expected columns cannot be detected.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info("Loading data from '%s' …", path)
        try:
            self._df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.exception("Failed to read CSV '%s'", path)
            raise RuntimeError(f"Could not read '{path}': {exc}") from exc

        logger.info("Raw shape: %s", self._df.shape)

        # Detect ID and label columns (case-insensitive)
        col_upper_map: Dict[str, str] = {c.upper(): c for c in self._df.columns}

        if self._ID_COL.upper() not in col_upper_map:
            raise ValueError(
                f"Could not find consumer-ID column (expected '{self._ID_COL}'). "
                f"Available columns: {list(self._df.columns)}"
            )
        if self._LABEL_COL.upper() not in col_upper_map:
            raise ValueError(
                f"Could not find label column (expected '{self._LABEL_COL}'). "
                f"Available columns: {list(self._df.columns)}"
            )

        self._id_col = col_upper_map[self._ID_COL.upper()]
        self._label_col = col_upper_map[self._LABEL_COL.upper()]
        logger.info(
            "Detected ID column: '%s', label column: '%s'",
            self._id_col, self._label_col,
        )

        # Coerce label to integer
        self._df[self._label_col] = (
            pd.to_numeric(self._df[self._label_col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        self._is_loaded = True

        # Coerce feature columns to float
        feat_cols = self.get_feature_columns()
        self._df[feat_cols] = self._df[feat_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        logger.info("Data loaded successfully.  Shape: %s", self._df.shape)
        return self

    # ------------------------------------------------------------------
    # validate_data
    # ------------------------------------------------------------------

    def validate_data(self) -> Dict:
        """
        Perform basic validation checks on the loaded DataFrame.

        Checks include:
          * Shape sanity (at least 1 row and 3 columns)
          * Label column cardinality
          * Per-column dtype summary
          * Per-column missing-value ratio

        Returns
        -------
        dict
            Dictionary with validation results.

        Raises
        ------
        RuntimeError
            If data has not been loaded.
        ValueError
            If validation fails on any critical check.
        """
        self._assert_loaded()
        df = self._df  # local alias

        report: Dict = {}

        # Shape
        n_rows, n_cols = df.shape
        report["n_rows"] = n_rows
        report["n_cols"] = n_cols
        logger.info("Shape: %d rows × %d columns", n_rows, n_cols)

        if n_rows < 1:
            raise ValueError("DataFrame has no rows after loading.")
        if n_cols < 3:
            raise ValueError(
                f"Expected at least 3 columns (id, label, ≥1 feature).  Got {n_cols}."
            )

        # Label cardinality
        label_vals = df[self._label_col].unique().tolist()
        report["label_unique_values"] = label_vals
        if not set(label_vals).issubset({0, 1}):
            logger.warning(
                "Label column contains unexpected values: %s", label_vals
            )

        # dtypes
        report["dtypes"] = df.dtypes.astype(str).to_dict()

        # Missing values
        missing_ratio: pd.Series = df.isnull().mean()
        report["missing_ratio"] = missing_ratio.to_dict()
        high_missing = missing_ratio[missing_ratio > 0.5]
        if not high_missing.empty:
            logger.warning(
                "Columns with >50%% missing data: %s",
                high_missing.index.tolist(),
            )
        else:
            logger.info("No columns with >50%% missing data.")

        overall_missing = df.isnull().mean().mean()
        report["overall_missing_ratio"] = float(overall_missing)
        logger.info(
            "Overall missing ratio: %.4f  (%.2f%%)",
            overall_missing, overall_missing * 100,
        )

        logger.info("Validation completed.")
        return report

    # ------------------------------------------------------------------
    # remove_duplicates
    # ------------------------------------------------------------------

    def remove_duplicates(self) -> "DataLoader":
        """
        Drop exact duplicate rows from the DataFrame (all columns must match).

        Returns
        -------
        self
        """
        self._assert_loaded()
        before = len(self._df)
        self._df = self._df.drop_duplicates()
        after = len(self._df)
        dropped = before - after
        if dropped:
            logger.info("Removed %d duplicate row(s).  Rows now: %d", dropped, after)
        else:
            logger.info("No duplicate rows found.")
        self._df = self._df.reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    # handle_missing_values
    # ------------------------------------------------------------------

    def handle_missing_values(
        self,
        strategy: str = "interpolate",
        fill_value: Optional[float] = None,
    ) -> "DataLoader":
        """
        Handle missing values in the feature columns.

        Parameters
        ----------
        strategy : str
            One of ``"interpolate"``, ``"mean"``, ``"median"``, ``"drop"``.
            Default ``"interpolate"``.
        fill_value : float, optional
            Used only when *strategy* is not one of the named ones (fallback
            constant fill).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If *strategy* is unrecognised.
        """
        self._assert_loaded()
        valid_strategies = {"interpolate", "mean", "median", "drop"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'.  Choose from {valid_strategies}."
            )

        feat_cols = self.get_feature_columns()
        before_missing = self._df[feat_cols].isnull().sum().sum()
        logger.info(
            "Handling %d missing value(s) using strategy='%s' …",
            before_missing, strategy,
        )

        if strategy == "interpolate":
            # Linear interpolation along each row (axis=1), then forward/back fill residual
            self._df[feat_cols] = (
                self._df[feat_cols]
                .interpolate(method="linear", axis=1, limit_direction="both")
                .ffill(axis=1)
                .bfill(axis=1)
            )
        elif strategy == "mean":
            col_means = self._df[feat_cols].mean()
            self._df[feat_cols] = self._df[feat_cols].fillna(col_means)
        elif strategy == "median":
            col_medians = self._df[feat_cols].median()
            self._df[feat_cols] = self._df[feat_cols].fillna(col_medians)
        elif strategy == "drop":
            rows_before = len(self._df)
            self._df = self._df.dropna(subset=feat_cols)
            self._df = self._df.reset_index(drop=True)
            logger.info(
                "Dropped %d row(s) with missing values.  Rows remaining: %d",
                rows_before - len(self._df), len(self._df),
            )

        after_missing = self._df[feat_cols].isnull().sum().sum()
        logger.info(
            "Missing values after handling: %d (was %d)", after_missing, before_missing
        )
        return self

    # ------------------------------------------------------------------
    # detect_and_handle_outliers
    # ------------------------------------------------------------------

    def detect_and_handle_outliers(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        isolation_forest_contamination: float = 0.05,
        action: str = "clip",
    ) -> "DataLoader":
        """
        Detect and handle outliers in the feature columns.

        Parameters
        ----------
        method : str
            One of ``"iqr"``, ``"zscore"``, ``"isolation_forest"``.
        threshold : float
            IQR multiplier (for ``"iqr"``) or z-score threshold (for ``"zscore"``).
            Ignored for ``"isolation_forest"``.
        isolation_forest_contamination : float
            Expected fraction of outlier rows (used by ``"isolation_forest"``).
        action : str
            What to do with detected outlier *values*:
            ``"clip"`` (default) – clip to boundary; ``"nan"`` – replace with NaN
            (call ``handle_missing_values`` afterwards).
            For ``"isolation_forest"`` entire *rows* are removed.

        Returns
        -------
        self
        """
        self._assert_loaded()
        valid_methods = {"iqr", "zscore", "isolation_forest"}
        if method not in valid_methods:
            raise ValueError(
                f"Unknown outlier method '{method}'.  Choose from {valid_methods}."
            )

        feat_cols = self.get_feature_columns()
        feat_data = self._df[feat_cols].copy()

        logger.info(
            "Detecting outliers with method='%s', threshold=%.2f …",
            method, threshold,
        )

        if method == "iqr":
            q1 = feat_data.quantile(0.25)
            q3 = feat_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            if action == "clip":
                feat_data = feat_data.clip(lower=lower, upper=upper, axis=1)
                logger.info("Clipped IQR outliers (multiplier=%.2f).", threshold)
            else:
                mask = (feat_data < lower) | (feat_data > upper)
                feat_data[mask] = np.nan
                logger.info(
                    "Replaced %d IQR outlier(s) with NaN.", int(mask.sum().sum())
                )

        elif method == "zscore":
            z = np.abs(stats.zscore(feat_data.fillna(0), axis=0, nan_policy="omit"))
            mask = z > threshold
            if action == "clip":
                mean_vals = feat_data.mean()
                std_vals = feat_data.std().replace(0, 1)
                lower_z = mean_vals - threshold * std_vals
                upper_z = mean_vals + threshold * std_vals
                feat_data = feat_data.clip(lower=lower_z, upper=upper_z, axis=1)
                logger.info(
                    "Clipped Z-score outliers (threshold=%.2f).", threshold
                )
            else:
                feat_data[mask] = np.nan
                logger.info(
                    "Replaced %d Z-score outlier(s) with NaN.",
                    int(mask.sum().sum()),
                )

        elif method == "isolation_forest":
            iso = IsolationForest(
                contamination=isolation_forest_contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            filled = feat_data.fillna(feat_data.median())
            preds = iso.fit_predict(filled)
            n_outlier_rows = int((preds == -1).sum())
            logger.info(
                "IsolationForest identified %d outlier row(s) (contamination=%.3f). "
                "Removing them.",
                n_outlier_rows, isolation_forest_contamination,
            )
            self._df = self._df[preds == 1].reset_index(drop=True)
            # feat_data already excluded; no in-place replacement needed
            logger.info("Rows remaining after outlier removal: %d", len(self._df))
            return self

        self._df[feat_cols] = feat_data
        return self

    # ------------------------------------------------------------------
    # get_feature_columns
    # ------------------------------------------------------------------

    def get_feature_columns(self) -> List[str]:
        """
        Return the list of feature column names (i.e., all columns that are
        neither the consumer-ID column nor the label column).

        Returns
        -------
        list of str

        Raises
        ------
        RuntimeError
            If data has not been loaded.
        """
        self._assert_loaded()
        exclude = {self._id_col, self._label_col}
        return [c for c in self._df.columns if c not in exclude]

    # ------------------------------------------------------------------
    # split_data
    # ------------------------------------------------------------------

    def split_data(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        stratify: bool = True,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train, validation, and test sets.

        The split is performed in two steps:
          1. Carve out *test_size* of the full dataset as test.
          2. From the remaining portion, carve out an adjusted fraction
             so that the validation set equals *val_size* of the **total**.

        Parameters
        ----------
        test_size : float
            Fraction of total samples allocated to the test set.
        val_size : float
            Fraction of total samples allocated to the validation set.
        stratify : bool
            Whether to stratify splits by the label column.
        random_state : int, optional
            Override random state for this call.

        Returns
        -------
        tuple of (train_df, val_df, test_df)

        Raises
        ------
        ValueError
            If split fractions are invalid.
        """
        self._assert_loaded()

        if test_size <= 0 or val_size <= 0:
            raise ValueError("test_size and val_size must be positive.")
        if test_size + val_size >= 1.0:
            raise ValueError(
                f"test_size + val_size = {test_size + val_size:.3f} ≥ 1.0."
            )

        rng = random_state if random_state is not None else self.random_state
        df = self._df.copy()
        stratify_col = df[self._label_col] if stratify else None

        # Step 1: separate test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=rng,
        )

        # Step 2: separate validation from train_val
        # val_size relative to full dataset → relative to remaining
        val_fraction_of_trainval = val_size / (1.0 - test_size)
        stratify_trainval = (
            train_val_df[self._label_col] if stratify else None
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_fraction_of_trainval,
            stratify=stratify_trainval,
            random_state=rng,
        )

        self._train = train_df.reset_index(drop=True)
        self._val = val_df.reset_index(drop=True)
        self._test = test_df.reset_index(drop=True)
        self._is_split = True

        logger.info(
            "Split complete → train: %d, val: %d, test: %d",
            len(self._train), len(self._val), len(self._test),
        )
        return self._train, self._val, self._test

    # ------------------------------------------------------------------
    # scale_features
    # ------------------------------------------------------------------

    def scale_features(
        self,
        scaler_type: str = "robust",
    ) -> "DataLoader":
        """
        Fit a scaler on the training split and transform train, val, and test.

        Parameters
        ----------
        scaler_type : str
            One of ``"standard"``, ``"robust"``, ``"minmax"``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If *scaler_type* is unrecognised.
        RuntimeError
            If data has not been split yet.
        """
        self._assert_split()

        scaler_type = scaler_type.lower()
        if scaler_type not in self._SCALER_MAP:
            raise ValueError(
                f"Unknown scaler '{scaler_type}'.  "
                f"Choose from {list(self._SCALER_MAP.keys())}."
            )

        feat_cols = self.get_feature_columns()
        scaler_cls = self._SCALER_MAP[scaler_type]
        self._scaler = scaler_cls()
        self._scaler_type = scaler_type

        logger.info("Fitting %s on training data …", scaler_cls.__name__)

        # Fit on train
        self._train[feat_cols] = self._scaler.fit_transform(
            self._train[feat_cols].values
        )
        # Transform val and test
        self._val[feat_cols] = self._scaler.transform(
            self._val[feat_cols].values
        )
        self._test[feat_cols] = self._scaler.transform(
            self._test[feat_cols].values
        )

        self._is_scaled = True
        logger.info("Scaling complete using %s.", scaler_cls.__name__)
        return self

    # ------------------------------------------------------------------
    # get_numpy_arrays
    # ------------------------------------------------------------------

    def get_numpy_arrays(
        self,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
    ]:
        """
        Return feature matrices and label vectors as NumPy arrays.

        Returns
        -------
        tuple
            ``(X_train, X_val, X_test, y_train, y_val, y_test)``

        Raises
        ------
        RuntimeError
            If data has not been split yet.
        """
        self._assert_split()

        feat_cols = self.get_feature_columns()

        X_train = self._train[feat_cols].values.astype(np.float32)
        X_val = self._val[feat_cols].values.astype(np.float32)
        X_test = self._test[feat_cols].values.astype(np.float32)

        y_train = self._train[self._label_col].values.astype(np.int64)
        y_val = self._val[self._label_col].values.astype(np.int64)
        y_test = self._test[self._label_col].values.astype(np.int64)

        logger.info(
            "Returning numpy arrays:  X_train %s, X_val %s, X_test %s",
            X_train.shape, X_val.shape, X_test.shape,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # get_data_summary
    # ------------------------------------------------------------------

    def get_data_summary(self) -> Dict:
        """
        Print and return a comprehensive summary of the current dataset state.

        Returns
        -------
        dict
            Keys include ``shape``, ``class_distribution``,
            ``missing_values``, ``feature_count``, ``scaler``,
            and split sizes if available.
        """
        self._assert_loaded()

        feat_cols = self.get_feature_columns()
        label_counts = self._df[self._label_col].value_counts().to_dict()
        total = len(self._df)
        missing_total = int(self._df[feat_cols].isnull().sum().sum())
        missing_per_col: Dict[str, float] = (
            self._df[feat_cols].isnull().mean().round(4).to_dict()
        )

        summary: Dict = {
            "shape": self._df.shape,
            "n_features": len(feat_cols),
            "feature_columns": feat_cols,
            "class_distribution": label_counts,
            "class_ratio": {
                k: round(v / total, 4) for k, v in label_counts.items()
            },
            "missing_values_total": missing_total,
            "missing_values_per_column": missing_per_col,
            "scaler": self._scaler_type,
            "is_split": self._is_split,
            "is_scaled": self._is_scaled,
        }

        if self._is_split:
            summary["train_size"] = len(self._train)
            summary["val_size"] = len(self._val)
            summary["test_size"] = len(self._test)

        # Pretty print
        print("=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"  Shape             : {summary['shape']}")
        print(f"  Feature columns   : {summary['n_features']}")
        print(f"  Missing values    : {missing_total}")
        print(f"  Class distribution: {label_counts}")
        print(f"  Class ratio       : {summary['class_ratio']}")
        print(f"  Scaler            : {self._scaler_type}")
        print(f"  Split performed   : {self._is_split}")
        if self._is_split:
            print(
                f"  Train/Val/Test    : "
                f"{summary['train_size']} / {summary['val_size']} / {summary['test_size']}"
            )
        print("=" * 60)

        logger.info("Data summary generated.")
        return summary

    # ------------------------------------------------------------------
    # save_processed / load_processed
    # ------------------------------------------------------------------

    def save_processed(self, path: Union[str, Path]) -> None:
        """
        Serialise the entire DataLoader state to a pickle file.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g., ``"processed/data.pkl"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving processed DataLoader to '%s' …", path)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved successfully (%.2f MB).", path.stat().st_size / 1e6)

    @classmethod
    def load_processed(cls, path: Union[str, Path]) -> "DataLoader":
        """
        Deserialise a previously saved DataLoader instance from disk.

        Parameters
        ----------
        path : str or Path
            Path to the pickle file created by :meth:`save_processed`.

        Returns
        -------
        DataLoader
            Restored instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Processed file not found: '{path}'")
        logger.info("Loading processed DataLoader from '%s' …", path)
        with open(path, "rb") as fh:
            instance = pickle.load(fh)
        if not isinstance(instance, cls):
            raise TypeError(
                f"Loaded object is of type '{type(instance).__name__}', expected 'DataLoader'."
            )
        logger.info("Loaded successfully.")
        return instance

    # ------------------------------------------------------------------
    # Properties (read-only access to splits)
    # ------------------------------------------------------------------

    @property
    def train(self) -> Optional[pd.DataFrame]:
        """Training split DataFrame."""
        return self._train

    @property
    def val(self) -> Optional[pd.DataFrame]:
        """Validation split DataFrame."""
        return self._val

    @property
    def test(self) -> Optional[pd.DataFrame]:
        """Test split DataFrame."""
        return self._test

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """The full (possibly pre-split) DataFrame."""
        return self._df

    @property
    def scaler(self) -> Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]]:
        """The fitted scaler instance (``None`` if not yet scaled)."""
        return self._scaler

    def __repr__(self) -> str:  # pragma: no cover
        loaded = f"shape={self._df.shape}" if self._is_loaded else "not loaded"
        return (
            f"DataLoader("
            f"random_state={self.random_state}, "
            f"data={loaded}, "
            f"split={self._is_split}, "
            f"scaled={self._is_scaled})"
        )
