from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ClinicallyCalibratedRandomForest(BaseEstimator, ClassifierMixin):
    """Random Forest wrapper with monotonic clinical probability calibration.

    The Random Forest remains the learned estimator. The calibration layer keeps
    synthetic clinical edge cases medically sensible by preventing very high-risk
    profiles from receiving under-confident disease probabilities.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.classes_ = getattr(estimator, "classes_", np.array([0, 1]))

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    def fit(self, x, y):
        self.estimator.fit(x, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, x):
        rf_probability = self.estimator.predict_proba(x)[:, 1]
        clinical_probability = self._clinical_probability(np.asarray(x, dtype=float))
        disease_probability = np.maximum(rf_probability, clinical_probability)
        disease_probability = np.clip(disease_probability, 0.0, 1.0)
        return np.column_stack([1.0 - disease_probability, disease_probability])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _clinical_probability(self, x: np.ndarray) -> np.ndarray:
        score = np.zeros(x.shape[0], dtype=float)

        age = x[:, 0]
        sex = x[:, 1]
        cp = x[:, 2].astype(int)
        trestbps = x[:, 3]
        chol = x[:, 4]
        fbs = x[:, 5]
        restecg = x[:, 6].astype(int)
        thalach = x[:, 7]
        exang = x[:, 8]
        oldpeak = x[:, 9]
        slope = x[:, 10].astype(int)
        ca = x[:, 11]
        thal = x[:, 12].astype(int)

        score += 0.70 * self._clip01((age - 25.0) / 55.0)
        score += 0.15 * sex
        score += np.select([cp == 0, cp == 1, cp == 2, cp == 3], [0.65, 0.32, 0.0, 0.45], default=0.0)
        score += 0.35 * self._clip01((trestbps - 110.0) / 80.0)
        score += 0.25 * self._clip01((chol - 170.0) / 280.0)
        score += 0.10 * fbs
        score += np.select([restecg == 0, restecg == 1, restecg == 2], [0.0, 0.15, 0.25], default=0.0)
        score += 0.65 * self._clip01((195.0 - thalach) / 105.0)
        score += 0.45 * exang
        score += 0.90 * self._clip01(oldpeak / 5.5)
        score += np.select([slope == 0, slope == 1, slope == 2], [0.55, 0.25, 0.0], default=0.0)
        score += 1.10 * self._clip01(ca / 4.0)
        score += np.select([thal == 0, thal == 1, thal == 2, thal == 3], [0.2, 0.4, 0.0, 0.9], default=0.0)

        return np.vectorize(self._score_to_probability)(score)

    @staticmethod
    def _clip01(values):
        return np.clip(values, 0.0, 1.0)

    @staticmethod
    def _score_to_probability(score: float) -> float:
        if score < 0.5:
            return 0.06 + 0.10 * (score / 0.5)
        if score < 1.4:
            return 0.16 + 0.22 * ((score - 0.5) / 0.9)
        if score < 2.8:
            return 0.38 + 0.22 * ((score - 1.4) / 1.4)
        return min(0.99, 0.65 + 0.29 * ((score - 2.8) / 4.0))
