"""Figures for the core-loss model comparison (matplotlib only)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from compare_n87 import MODELS

mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.direction": "out", "ytick.direction": "out", "savefig.dpi": 300})

LABEL = {"SE (sine-fit)": "SE (sine fit)", "MSE (sine-fit)": "MSE (sine fit)",
         "iGSE (sine-fit)": "iGSE (sine fit)", "iGSE (tri-fit)": "iGSE (triangle fit)",
         "i2GSE (tri-fit)": "i²GSE (triangle fit)",
         "iGSE+SPG (sine-fit per Hdc)": "iGSE + SPG (sine fit per H$_{dc}$)",
         "i2GSE+SPG (tri-fit per Hdc)": "i²GSE + SPG (triangle fit per H$_{dc}$)",
         "Composite loss-map (tri D=0.5 per Hdc)": "Composite loss map (triangles per H$_{dc}$)"}
REG7 = ["TX: sine", "TX: square-wave flux (tri D=0.5)", "TX: trapezoid dP=dN",
        "L: tri D≠0.5, no bias", "L: tri D≠0.5 + DC bias", "O: asym. trapezoid", "O: trapezoid + DC bias"]
REG7_LABEL = ["Sine\n(resonant)", "Square-wave flux\n(tri D=0.5)", "Trapezoid\nd$_P$=d$_N$",
              "Triangle D≠0.5\nno bias", "Triangle D≠0.5\n+ DC bias", "Asym.\ntrapezoid",
              "Trapezoid\n+ DC bias"]
COLOR = {"SE (sine-fit)": "0.6", "iGSE (sine-fit)": "#4477AA", "i2GSE (tri-fit)": "#66CCEE",
         "iGSE+SPG (sine-fit per Hdc)": "#CCBB44", "i2GSE+SPG (tri-fit per Hdc)": "#EE6677",
         "Composite loss-map (tri D=0.5 per Hdc)": "#AA3377"}
MARKER = {"SE (sine-fit)": "o", "iGSE (sine-fit)": "s", "i2GSE (tri-fit)": "^",
          "iGSE+SPG (sine-fit per Hdc)": "D", "i2GSE+SPG (tri-fit per Hdc)": "v",
          "Composite loss-map (tri D=0.5 per Hdc)": "P"}


def _letter(ax, s):
    ax.text(-0.08, 1.13, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def heatmap(R: pd.DataFrame, path):
    """Median |error| and median signed error, models x regimes; hatched = in-sample."""
    M1 = R.pivot(index="model", columns="regime", values="med_abs_err_pct").loc[MODELS, REG7].values
    M2 = R.pivot(index="model", columns="regime", values="median_bias_pct").loc[MODELS, REG7].values
    INS = R.pivot(index="model", columns="regime", values="in_sample").loc[MODELS, REG7].values.astype(bool)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True, gridspec_kw=dict(wspace=0.32))
    for ax, M, cm, norm, ttl, cl in [
            (axs[0], M1, "viridis_r", mpl.colors.Normalize(0, 50), "Median |error| (%)", "a"),
            (axs[1], M2, "RdBu_r", TwoSlopeNorm(0, -50, 50),
             "Median signed error (%)\n(− = model underestimates)", "b")]:
        im = ax.imshow(M, cmap=cm, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if INS[i, j]:
                    ax.add_patch(mpl.patches.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                       hatch="///", ec="0.7", lw=0))
                v = M[i, j]; rgba = im.cmap(im.norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                s = (f"{v:+.0f}" if round(v) != 0 else "0") if cl == "b" else f"{v:.0f}"
                ax.text(j, i, s, ha="center", va="center", fontsize=6, color="k" if lum > 0.5 else "w",
                        bbox=dict(boxstyle="square,pad=0.1", fc=rgba, ec="none") if INS[i, j] else None)
        ax.set_xticks(range(len(REG7))); ax.set_xticklabels(REG7_LABEL, rotation=90)
        ax.tick_params(axis="x", length=0)
        for x in (2.5, 4.5):
            ax.axvline(x, color="w", lw=2)
        for x, t in ((1, "Transformer"), (3.5, "Inductor"), (5.5, "Other")):
            ax.text(x, -0.75, t, ha="center", va="bottom", fontsize=7)
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=6); cb.set_label(ttl, fontsize=7)
        _letter(ax, cl)
    axs[0].set_yticks(range(len(MODELS))); axs[0].set_yticklabels([LABEL[m] for m in MODELS], fontsize=7)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def mechanisms(A: pd.DataFrame, D, pred, path):
    """Median signed error vs triangle duty, DC bias, and trapezoid dwell fraction."""
    et = A.excitation_type.values; Hv = A.Hdc.values; keep = Hv <= 45
    dpc = A.dPc.values; z = np.round(A.duty_zero_each.values, 1)
    panels = [
        (keep & (et == "triangular") & (Hv == 0), dpc, "Triangle duty D\n(H$_{dc}$ = 0)",
         "Short edges defeat\na single-α power law"),
        (keep & (et == "triangular") & (np.abs(dpc - 0.5) > 1e-6), Hv,
         "DC bias H$_{dc}$ (A/m)\n(triangles, D ≠ 0.5)", "Ignoring DC bias\nunderestimates loss"),
        (keep & (A.regime.values == "TX: trapezoid dP=dN"), z,
         "Dwell fraction per interval d$_0$\n(symmetric trapezoid, H$_{dc}$ = 0)",
         "Dwell intervals add\nloss that iGSE misses")]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (m, key, xl, tt), cl in zip(axs, panels, "abc"):
        ax.axhline(0, color="k", lw=0.6); ax.axhspan(-10, 10, color="0.93", zorder=0)
        for nm in COLOR:
            e = pd.Series(pred[nm][m] / D.P[m] - 1).groupby(key[m]).median() * 100
            ax.plot(e.index, e.values, marker=MARKER[nm], ms=3.5, lw=1.2, color=COLOR[nm],
                    label=LABEL[nm], zorder=3 if ("SPG" in nm or "Composite" in nm) else 2)
        ax.set_xlabel(xl); ax.set_title(tt, loc="left", pad=4); _letter(ax, cl); ax.margins(x=0.05)
    axs[0].set_ylabel("Median signed error (%)"); axs[0].set_ylim(-80, 32)
    axs[1].set_xticks([0, 15, 30, 45])
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.2))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
