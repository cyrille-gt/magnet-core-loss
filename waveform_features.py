"""Waveform features needed by Steinmetz-family core-loss models, computed on measured B(t).

Per measurement (1024-sample period, circular):
    dBpp      peak-to-peak flux density ΔB [T]
    ms2       <(dB/dt)^2> over the period                       (MSE)
    G<a>      <|dB/dt|^a> for a on ALPHA_GRID                   (iGSE / i2GSE)
    s0..s3    mean slope of each piecewise-linear segment [T/s] (i2GSE, composite)
    l0..l3    segment duration as fraction of the period
    seg_r2    fraction of slope energy explained by the piecewise-constant-slope fit
Triangles: 2 segments (rise dP, fall 1-dP).
Trapezoids: 4 segments rise dP / dwell d0 / fall dN / dwell d0; segment phase found
by a circular-shift search that maximises explained slope energy.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

NSAMP = 1024
ALPHA_GRID = np.round(np.arange(0.8, 3.001, 0.02), 3)
G_COLS = [f"G{a:.2f}" for a in ALPHA_GRID]


def segment_fit(d: np.ndarray, lens: np.ndarray, step: int = 2):
    """d: (n,1024) per-sample ΔB; lens: (n,K) integer segment lengths summing to 1024.
    Returns (best offset, per-segment ΔB sums (n,K))."""
    n, K = lens.shape
    C = np.concatenate([np.zeros((n, 1)), np.cumsum(np.concatenate([d, d], 1), 1)], 1)
    offs = np.arange(0, NSAMP, step)
    bnd = offs[None, :, None] + np.concatenate([np.zeros((n, 1), int), np.cumsum(lens, 1)], 1)[:, None, :]
    vals = np.take_along_axis(C, bnd.reshape(n, -1), axis=1).reshape(n, len(offs), K + 1)
    sums = np.diff(vals, axis=2)
    score = (sums ** 2 / np.maximum(lens[:, None, :], 1)).sum(2)
    best = score.argmax(1)
    return offs[best], sums[np.arange(n), best]


def waveform_features(B: np.ndarray, scalar_rows: pd.DataFrame) -> pd.DataFrame:
    """B: (n,1024) flux density [T]; scalar_rows: matching scalar-table rows (same order)."""
    f = scalar_rows.frequency_hz.values
    d = np.diff(np.concatenate([B, B[:, :1]], 1), axis=1)
    sl = np.abs(d) * NSAMP * f[:, None]
    feat = pd.DataFrame({"measurement_id": scalar_rows.measurement_id.values,
                         "dBpp": B.max(1) - B.min(1), "ms2": np.mean(sl ** 2, 1)})
    et = scalar_rows.excitation_type.values
    n = len(B)
    S = np.full((n, 4), np.nan); L = np.full((n, 4), np.nan); R2 = np.full(n, np.nan)
    for typ, K in (("triangular", 2), ("trapezoidal", 4)):
        ii = np.where(et == typ)[0]
        if len(ii) == 0:
            continue
        dp = np.rint(scalar_rows.duty_positive.values[ii] * NSAMP).astype(int)
        dn = np.rint(scalar_rows.duty_negative.values[ii] * NSAMP).astype(int)
        if K == 2:
            lens = np.stack([dp, NSAMP - dp], 1)
        else:
            z = (NSAMP - dp - dn) // 2
            lens = np.stack([dp, z, dn, NSAMP - dp - dn - z], 1)
        _, sums = segment_fit(d[ii], lens)
        S[ii, :K] = sums / lens * NSAMP * f[ii, None]
        L[ii, :K] = lens / NSAMP
        R2[ii] = (sums ** 2 / lens).sum(1) / (d[ii] ** 2).sum(1)
    for j in range(4):
        feat[f"s{j}"] = S[:, j]
        feat[f"l{j}"] = L[:, j]
    feat["seg_r2"] = R2
    G = np.stack([np.mean(sl ** a, 1) for a in ALPHA_GRID], 1)
    return pd.concat([feat, pd.DataFrame(G, columns=G_COLS)], axis=1)


def extract_material(material: str, scalar: pd.DataFrame, parts=None, verbose=True) -> pd.DataFrame:
    """Features for every measurement of one material (≈12 min for N87's 145k rows)."""
    from magnet_io import iter_waveforms
    sidx = scalar.set_index("measurement_id")
    out, t0 = [], time.time()
    for ids, B in iter_waveforms(material, scalar, parts):
        rows = sidx.loc[ids].reset_index()
        out.append(waveform_features(B, rows))
        if verbose:
            print(f"{sum(len(o) for o in out)} rows, {time.time() - t0:.0f} s", end="\r")
    return pd.concat(out, ignore_index=True)
