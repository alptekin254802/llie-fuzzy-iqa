"""Product t-norm and ordered singleton consequents for the NR-mono variant.

Over each transition interval, a Ruspini partition interpolates two adjacent
consequents with nonnegative weights. Ordering the 54 adjacent pairs therefore
makes the response nondecreasing in each oriented feature. Outer shoulders are
constant. The guarantee concerns the response, not agreement with observers.
"""
import numpy as np
from scipy.optimize import minimize, nnls
from threadpoolctl import threadpool_limits
from evaluation.frozen_scorer import RULES, FrozenScorer

INDEX = {tuple(v): i for i, v in enumerate(RULES)}
PAIRS = [(i, INDEX[tuple(v + np.eye(3, dtype=int)[axis])])
         for i, v in enumerate(RULES) for axis in range(3) if v[axis] < 2]
C = np.zeros((len(PAIRS), 27))
for row, (a, b) in zip(C, PAIRS):
    row[a], row[b] = -1, 1


def memberships(x, mfs):
    x = np.asarray(x, float)[:, :, None]
    a, b, c, d = [np.asarray(mfs)[..., i] for i in range(4)]
    left = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1)
    right = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1)
    return np.clip(np.minimum(np.minimum(left, 1), right), 0, 1)


def design(x, mfs):
    mu = memberships(x, mfs)
    # Fix summation layout: near-tied ranks are sensitive to last-bit changes.
    f = np.ascontiguousarray(mu[:, 0, RULES[:, 0]] * mu[:, 1, RULES[:, 1]] * mu[:, 2, RULES[:, 2]])
    return f / (f.sum(axis=1, keepdims=True) + 1e-12)


def fit(oriented_features, target, mfs):
    """Use one BLAS thread for the canonical constrained fit."""
    with threadpool_limits(limits=1):
        return _fit(oriented_features, target, mfs)


def _fit(oriented_features, target, mfs):
    """Fit constrained least squares; certify feasibility and KKT stationarity.

    Membership breakpoints are supplied from the intended calibration partition.
    SLSQP can warn near an optimum; a numerical certificate is checked separately.
    Predictions near tied plateaus remain sensitive to numerical precision.
    """
    f = design(oriented_features, mfs)
    y = np.asarray(target, dtype=float)
    constraints = [{"type": "ineq", "fun": lambda th, a=a, b=b: th[b] - th[a]} for a, b in PAIRS]
    result = minimize(lambda th: np.sum((f @ th - y)**2), np.full(27, y.mean()),
                      jac=lambda th: 2 * f.T @ (f @ th - y), constraints=constraints,
                      method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
    slack = C @ result.x
    grad = 2 * f.T @ (f @ result.x - y) / len(y)
    active = C[np.abs(slack) <= 1e-7].T
    multipliers, _ = nnls(active, grad, maxiter=10000)
    residual = float(np.max(np.abs(grad - active @ multipliers)))
    if np.min(slack) < -1e-8 or residual > 1e-6:
        raise RuntimeError(f"Constrained fit failed numerical certificate: {result.message}; residual={residual}")
    return result.x


def frozen_model():
    return FrozenScorer("nr_mono")
