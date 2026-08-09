"""Statistics helpers.

Two things drive the choices here.

First, probe items are **not independent**: several items come from the same
ToolShed task and therefore share a goal, a system prompt and a prefix. Treating
them as independent would shrink confidence intervals by roughly sqrt(items per
task). Every interval below is a cluster bootstrap, resampling *tasks* with
replacement and taking all of a task's items along with it.

Second, all our headline quantities are **paired differences of the same string
under two contexts**, so the natural tests are paired ones and the natural
effect size is Cohen's d_z. We report both the effect size and the interval;
p-values alone would be uninformative at n in the hundreds, where trivially
small differences are "significant".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int
    n_clusters: int
    #: Cohen's d_z for a paired difference (i.e. mean/sd of the differences).
    dz: Optional[float] = None
    #: Two-sided p-value from a cluster bootstrap of the mean (H0: mean = 0).
    p: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "n": self.n,
            "n_clusters": self.n_clusters,
            "dz": self.dz,
            "p": self.p,
        }

    def fmt(self, digits: int = 2) -> str:
        return f"{self.mean:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 20260808,
    statistic: str = "mean",
) -> Estimate:
    """Cluster bootstrap CI (and bootstrap p-value against zero) for a mean."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, 0)
    c = np.asarray(clusters)
    uniq, inverse = np.unique(c, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]

    def stat(x: np.ndarray) -> float:
        return float(np.median(x)) if statistic == "median" else float(np.mean(x))

    point = stat(v)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n_groups = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([groups[j] for j in pick])
        draws[b] = stat(v[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    # Two-sided bootstrap p-value: how far the null (0) sits in the bootstrap
    # distribution recentred on the point estimate.
    centred = draws - point
    p = float(2 * min((centred <= -abs(point)).mean(), (centred >= abs(point)).mean()))
    p = min(1.0, max(p, 1.0 / n_boot))
    sd = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
    dz = point / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
    return Estimate(point, float(lo), float(hi), int(v.size), n_groups, dz, p)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 20260808,
) -> Estimate:
    """Cluster bootstrap of mean(a - b) for paired observations."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return cluster_bootstrap(a - b, clusters, n_boot=n_boot, alpha=alpha, seed=seed)


def wilcoxon_p(a: Sequence[float], b: Optional[Sequence[float]] = None) -> float:
    """Wilcoxon signed-rank p-value (paired if `b` given). Falls back to NaN."""
    try:
        from scipy.stats import wilcoxon

        d = np.asarray(a, dtype=float)
        if b is not None:
            d = d - np.asarray(b, dtype=float)
        d = d[np.isfinite(d)]
        if d.size < 5 or np.allclose(d, 0):
            return float("nan")
        return float(wilcoxon(d).pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values, preserving the input keys."""
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {k: float("nan") for k in pvalues}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def bootstrap_proportion(
    successes: Sequence[bool],
    clusters: Sequence[str],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 20260808,
) -> Estimate:
    return cluster_bootstrap(
        [1.0 if s else 0.0 for s in successes], clusters, n_boot, alpha, seed
    )


def weighted_least_squares(
    x: Sequence[float], y: Sequence[float], w: Optional[Sequence[float]] = None
) -> Tuple[float, float, float]:
    """Return (slope, intercept, r2) for a simple (optionally weighted) fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    W = np.diag(w)
    X = np.vstack([x, np.ones_like(x)]).T
    beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
    pred = X @ beta
    ss_res = float(((y - pred) ** 2 * w).sum())
    ss_tot = float(((y - np.average(y, weights=w)) ** 2 * w).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(beta[0]), float(beta[1]), r2


def crossing_point(
    x: Sequence[float], y: Sequence[float]
) -> Optional[float]:
    """Where a monotone-ish fit of y(x) crosses zero, by linear interpolation."""
    slope, intercept, _ = weighted_least_squares(x, y)
    if slope == 0:
        return None
    return float(-intercept / slope)
