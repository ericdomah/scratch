import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    Handles cleaning, normalization, and windowing for electricity consumption data.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()

    def interpolate_missing(self, series):
        """Pattern: Linear interpolation for missing consumption values"""
        series = pd.Series(series)
        return series.interpolate(method='linear').fillna(0).values

    def normalize(self, data):
        """Pattern: Min-Max scaling to [0, 1] range per consumer"""
        # Reshape for scaler
        data_reshaped = data.reshape(-1, 1)
        normalized = self.scaler.fit_transform(data_reshaped)
        return normalized.flatten()

    def detect_outliers(self, data, threshold=3):
        """Pattern: Simple Z-score outlier detection (clipping)"""
        data = data.copy() # Ensure we are not modifying a read-only view
        mean = np.mean(data)
        std = np.std(data)
        if std == 0: return data
        z_scores = (data - mean) / std
        # Clip outliers to thresholds
        data[z_scores > threshold] = mean + threshold * std
        data[z_scores < -threshold] = mean - threshold * std
        return data

    def process_user_data(self, consumption):
        """Full pipeline for a single user's time series"""
        clean = self.interpolate_missing(consumption)
        clipped = self.detect_outliers(clean)
        normed = self.normalize(clipped)
        return normed

if __name__ == "__main__":
    # Test Preprocessing
    sample = np.array([10.0, np.nan, 12.0, 500.0, 11.0]) # 500 is an outlier
    preprocessor = DataPreprocessor()
    processed = preprocessor.process_user_data(sample)
    print(f"Original: {sample}")
    print(f"Processed: {processed}")
