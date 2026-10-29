"""Frozen zero-order Sugeno models; input order: entropy, contrast, Laplacian energy."""
from pathlib import Path
import itertools
import json
import numpy as np

FEATURES = ("entropy", "contrast", "sharpness")
RULES = np.array(list(itertools.product(range(3), repeat=3)))
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class FrozenScorer:
    def __init__(self, model="nr_free"):
        path = MODEL_DIR / f"{model}.json" if isinstance(model, str) else None
        self.parameters = json.loads(path.read_text()) if path else model
        p = self.parameters
        self.signs = np.array([-1.0 if p["signs"][k] else 1.0 for k in FEATURES])
        self.mfs = np.asarray(p["mfs"], dtype=float)
        self.theta = np.asarray(p["theta"], dtype=float)
        self.product = p["t_norm"] == "product"
        self.epsilon = 1e-12 if self.product else 1e-9
        self.a, self.b, self.c, self.d = [self.mfs[..., i] for i in range(4)]

    def __call__(self, features):
        x = (self.signs * np.asarray(features, dtype=float))[:, None]
        left = np.where(self.b > self.a, (x - self.a) / np.maximum(self.b - self.a, 1e-12), 1)
        right = np.where(self.d > self.c, (self.d - x) / np.maximum(self.d - self.c, 1e-12), 1)
        mu = np.clip(np.minimum(np.minimum(left, 1), right), 0, 1)
        terms = [mu[n, RULES[:, n]] for n in range(3)]
        w = terms[0] * terms[1] * terms[2] if self.product else np.minimum.reduce(terms)
        return float(w @ self.theta / (w.sum() + self.epsilon))

    def predict(self, features):
        return np.array([self(row) for row in np.asarray(features, dtype=float)])
