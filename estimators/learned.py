import numpy as np
import xgboost as xgb
from .base import Estimator
from .features import featurize, featurize_batch

# Learned cardinality estimator.
class LearnedEstimator(Estimator):
    name = "learned"
    # Initialize XGBoost model.
    def __init__(self, hist_estimator, **xgb_params):
        self.hist_estimator = hist_estimator
        params = dict(n_estimators=200, max_depth=6, learning_rate=0.08,
                       objective="reg:squarederror", subsample=0.9, colsample_bytree=0.9,
                       random_state=42)
        params.update(xgb_params)
        self.model = xgb.XGBRegressor(**params)
        self._fitted = False
    # Train the estimator.
    def fit(self, train_queries, train_true_cards):
        X = featurize_batch(train_queries, self.hist_estimator)
        y = np.log1p(np.maximum(np.array(train_true_cards, dtype=np.float64), 0))
        self.model.fit(X, y)
        self._fitted = True
    # Predict query cardinality.
    def estimate(self, query) -> float:
        if not self._fitted:
            raise RuntimeError("LearnedEstimator.fit() must be called before estimate()")
        x = featurize(query, self.hist_estimator).reshape(1, -1)
        log_pred = self.model.predict(x)[0]
        return max(float(np.expm1(log_pred)), 0.0)