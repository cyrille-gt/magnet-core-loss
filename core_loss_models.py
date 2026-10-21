"""Steinmetz-family core-loss models evaluated on measured waveforms (MagNet features).

Models (P in W/m^3, f in Hz, B in T):
    SE      P = k f^a Bpk^b
    MSE     P = k f_eq^(a-1) Bpk^b f,  f_eq = 2/(ΔB^2 π^2) ∫(dB/dt)^2 dt          (Reinert 1999)
    iGSE    P = ki ΔB^(b-a) <|dB/dt|^a>,  ki = k / ((2π)^(a-1) ∫|cosθ|^a dθ 2^(b-a))  (Venkatachalam 2002)
    i2GSE   iGSE + Σ_l Qr_l Pr_l  (Mühlethaler 2012)
            Pr_l = f kr |dB/dt(t-)|^ar ΔB^br (1 - exp(-t_l/τ)),  Qr_l = exp(-qr |dB/dt(t+)/dB/dt(t-)|)
            evaluated at every segment transition l; t_l = duration of the following segment.
    SPG     Steinmetz pre-magnetisation graph idea: (k, a, b) refitted at each H_dc level.
    Composite loss map: measured symmetric-triangle loss surface per (T, H_dc),
            log P = quadratic(log f, log Bpk), clamped-linear extrapolation; each segment j
            contributes E_sym(f_eq,j = |s_j|/(2ΔB), Bpk) * |s_j| t_j / (2ΔB).
            (Own implementation in the spirit of the composite waveform hypothesis.)

All fits minimise squared log error (i.e. relative error).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares

from waveform_features import ALPHA_GRID, G_COLS, NSAMP


class CoreLossData:
    """Column arrays of a merged scalar + feature table, indexed by boolean masks."""

    def __init__(self, A: pd.DataFrame):
        self.A = A
        self.f = A.frequency_hz.values
        self.dBpp = A.dBpp.values
        self.Bpk = self.dBpp / 2
        self.ms2 = A.ms2.values
        self.P = A.volumetric_loss_w_per_m3.values
        self.logG = np.log(A[G_COLS].values)
        self.S4 = A[["s0", "s1", "s2", "s3"]].values
        self.L4 = A[["l0", "l1", "l2", "l3"]].values
        et = A.excitation_type.values
        self.K = np.where(et == "trapezoidal", 4, np.where(et == "triangular", 2, 0))


_ki_cache: dict = {}


def ki_ratio(a: float, b: float) -> float:
    """ki / k for the iGSE."""
    key = (round(a, 6), round(b, 6))
    if key not in _ki_cache:
        I = quad(lambda th: abs(np.cos(th)) ** a, 0, 2 * np.pi)[0]
        _ki_cache[key] = 1 / ((2 * np.pi) ** (a - 1) * I * 2 ** (b - a))
    return _ki_cache[key]


def P_SE(p, D: CoreLossData, m):
    lk, a, b = p
    return np.exp(lk + a * np.log(D.f[m]) + b * np.log(D.Bpk[m]))


def P_MSE(p, D: CoreLossData, m):
    lk, a, b = p
    f = D.f[m]
    feq = 2 * D.ms2[m] / (D.dBpp[m] ** 2 * np.pi ** 2 * f)
    return np.exp(lk) * feq ** (a - 1) * D.Bpk[m] ** b * f


def P_iGSE(p, D: CoreLossData, m):
    lk, a, b = p
    x = (a - ALPHA_GRID[0]) / 0.02
    j = int(np.clip(np.floor(x), 0, len(ALPHA_GRID) - 2)); w = x - j
    lg = D.logG[m, j] * (1 - w) + D.logG[m, j + 1] * w
    return np.exp(lk) * ki_ratio(a, b) * np.exp((b - a) * np.log(D.dBpp[m]) + lg)


def fit_steinmetz(model, D: CoreLossData, m, p0=(1.0, 1.4, 2.5)):
    """Fit (ln k, alpha, beta) of SE / MSE / iGSE on rows m."""
    y = np.log(D.P[m])
    r = least_squares(lambda p: np.log(model(p, D, m)) - y, p0,
                      bounds=([-50, 0.85, 1.0], [50, 2.95, 4.0]))
    return r.x


def P_transition(q, D: CoreLossData, m):
    """i2GSE transition ('relaxation') term; q = (ln kr, ar, br, ln tau, qr)."""
    lkr, ar, br, ltau, qr = q
    f = D.f[m]; dB = D.dBpp[m]; S = D.S4[m]; L = D.L4[m]; K = D.K[m]
    r = np.arange(len(K)); out = np.zeros(len(K))
    for j in range(4):
        valid = j < K
        jn = np.where(K == 4, (j + 1) % 4, (j + 1) % 2)
        sm = np.abs(S[r, j])
        sp = np.abs(S[r, np.where(K > 0, jn, 0)])
        tl = L[r, np.where(K > 0, jn, 0)] / f
        with np.errstate(invalid="ignore", divide="ignore"):
            term = (np.exp(-qr * sp / sm) * f * np.exp(lkr) * sm ** ar * dB ** br
                    * (1 - np.exp(-tl / np.exp(ltau))))
        out += np.where(valid, np.nan_to_num(term), 0)
    return out


RELAX_STARTS = [[np.log(1e-6), 1.2, 2.5, np.log(2e-6), 5.0],
                [np.log(1e-12), 2.0, 1.0, np.log(1e-7), 2.0],
                [np.log(1e-3), 1.0, 3.0, np.log(1e-5), 10.0],
                [np.log(1e-9), 1.5, 2.0, np.log(5e-7), 16.0]]


def fit_transition(D: CoreLossData, m, base, tau=None):
    """Fit the i2GSE term on rows m given base-model loss `base`; multi-start.
    tau=None fits tau freely, otherwise holds it fixed (for identifiability profiles).
    Returns the least_squares result with the best cost."""
    y = np.log(D.P[m]); best = None
    for s0 in RELAX_STARTS:
        if tau is None:
            fun = lambda q: np.log(base + P_transition(q, D, m)) - y
            x0, lb, ub = s0, [-80, 0.0, 0.0, np.log(1e-9), 0], [30, 3.5, 5, np.log(1e-4), 50]
        else:
            fun = lambda q: np.log(base + P_transition(np.r_[q[:3], np.log(tau), q[3]], D, m)) - y
            x0, lb, ub = [s0[0], s0[1], s0[2], s0[4]], [-80, 0.0, 0.0, 0], [30, 3.5, 5, 50]
        try:
            r = least_squares(fun, x0, bounds=(lb, ub))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


# ---------------- composite loss map ----------------
def _design(x, y):
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1)


def fit_loss_map(f, Bpk, P):
    """Quadratic log-surface of symmetric-triangle loss; returns (coef, xmin, xmax, ymin, ymax)."""
    x, y = np.log(f), np.log(Bpk)
    c, *_ = np.linalg.lstsq(_design(x, y), np.log(P), rcond=None)
    return c, x.min(), x.max(), y.min(), y.max()


def loss_map_logP(lm, x, y):
    x, y = np.broadcast_arrays(x, y)
    c, x0, x1, y0, y1 = lm
    xc, yc = np.clip(x, x0, x1), np.clip(y, y0, y1)
    base = _design(xc, yc) @ c
    dx = c[1] + 2 * c[3] * xc + c[5] * yc
    dy = c[2] + 2 * c[4] * yc + c[5] * xc
    return base + dx * (x - xc) + dy * (y - yc)


def P_composite(lm, D: CoreLossData, m):
    ii = np.where(m)[0]
    f = D.f[ii]; dB = D.dBpp[ii]; y = np.log(D.Bpk[ii]); seg = D.K[ii] > 0
    tot = np.zeros(len(ii))
    for j in range(4):
        s = np.abs(D.S4[ii, j]); l = D.L4[ii, j]
        ok = seg & ~np.isnan(s) & (s > 0)
        s = np.where(ok, s, 1.0); l = np.where(ok, l, 0.0)
        feq = s / (2 * dB)
        tot += np.where(ok, np.exp(loss_map_logP(lm, np.log(feq), y)) / feq * s * l / f / (2 * dB), 0)
    sn = ~seg
    if sn.any():  # sinusoidal rows: analytic pure sine
        th = np.linspace(0, 2 * np.pi, NSAMP, endpoint=False)
        s = np.abs(np.cos(th))[None, :] * 2 * np.pi * f[sn, None] * D.Bpk[ii][sn, None]
        feq = np.maximum(s / (2 * dB[sn, None]), 1e-3)
        E = (np.exp(loss_map_logP(lm, np.log(feq), y[sn, None])) / feq * s
             * (1 / (f[sn, None] * NSAMP)) / (2 * dB[sn, None]))
        tot[sn] = E.sum(1)
    return tot * f
