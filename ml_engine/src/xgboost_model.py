import xgboost as xgb
import pickle

class XGBoostBaseline:
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss"
        )
    
    def _flatten(self, X):
        """Flattens (batch, seq_len, features) into (batch, seq_len * features)"""
        # If input is a torch tensor, move it to CPU and convert to numpy
        if hasattr(X, 'detach'):
            X = X.detach().cpu().numpy()
        
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def train(self, X_train, y_train):
        X_train_flat = self._flatten(X_train)
        if hasattr(y_train, 'detach'):
            y_train = y_train.detach().cpu().numpy()
        self.model.fit(X_train_flat, y_train)

    def predict_proba(self, X):
        X_flat = self._flatten(X)
        return self.model.predict_proba(X_flat)[:, 1]

    def predict(self, X):
        X_flat = self._flatten(X)
        return self.model.predict(X_flat)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
