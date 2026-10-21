"""Inductor vs transformer core-loss model comparison on MagNet N87.

Usage:
    python compare_n87.py [--material N87] [--out results] [--refeature] [--no-figures]

Pipeline:
 1. scalar table + waveform features (cached in D:\\Claude\\magnet\\derived\\<mat>_features.pkl);
 2. quality filter: quality_accepted & duty_quality_accepted & loss > 0;
 3. per-temperature fits:
      sine fit     (k,a,b) on sinusoidal, H_dc = 0            -> SE, MSE, iGSE
      triangle fit (k,a,b) on symmetric triangles, H_dc = 0  -> iGSE, i2GSE base
      i2GSE term   on H_dc = 0 trapezoids, 2-fold CV grouped by (dP,dN) command combo
      SPG          refit (k,a,b) per H_dc level (sine or symmetric triangle)
      composite    symmetric-triangle loss map per (T, H_dc)
 4. errors per regime (H_dc <= 45 A/m), in-sample cells flagged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import core_loss_models as clm
from magnet_io import DERIVED_ROOT, load_scalar
from waveform_features import extract_material

MODELS = ["SE (sine-fit)", "MSE (sine-fit)", "iGSE (sine-fit)", "iGSE (tri-fit)", "i2GSE (tri-fit)",
          "iGSE+SPG (sine-fit per Hdc)", "i2GSE+SPG (tri-fit per Hdc)",
          "Composite loss-map (tri D=0.5 per Hdc)"]
REGIMES = ["TX: sine", "TX: square-wave flux (tri D=0.5)", "TX: trapezoid dP=dN",
           "L: tri D≠0.5, no bias", "L: tri D≠0.5 + DC bias", "O: asym. trapezoid",
           "O: trapezoid + DC bias", "O: sine + DC bias", "O: tri D=0.5 + DC bias"]
IN_SAMPLE = {
    "SE (sine-fit)": ["TX: sine"], "MSE (sine-fit)": ["TX: sine"], "iGSE (sine-fit)": ["TX: sine"],
    "iGSE (tri-fit)": ["TX: square-wave flux (tri D=0.5)"],
    "i2GSE (tri-fit)": ["TX: square-wave flux (tri D=0.5)"],
    "iGSE+SPG (sine-fit per Hdc)": ["TX: sine", "O: sine + DC bias"],
    "i2GSE+SPG (tri-fit per Hdc)": ["TX: square-wave flux (tri D=0.5)", "O: tri D=0.5 + DC bias"],
    "Composite loss-map (tri D=0.5 per Hdc)": ["TX: square-wave flux (tri D=0.5)", "O: tri D=0.5 + DC bias"],
}
HDC_MAX = 45  # H_dc = 60 A/m too sparse (11 symmetric triangles per temperature for N87)


def regime(e, dp, dn, h):
    if e == "sinusoidal":
        return "TX: sine" if h == 0 else "O: sine + DC bias"
    if e == "triangular":
        if abs(dp - 0.5) < 1e-6:
            return "TX: square-wave flux (tri D=0.5)" if h == 0 else "O: tri D=0.5 + DC bias"
        return "L: tri D≠0.5, no bias" if h == 0 else "L: tri D≠0.5 + DC bias"
    if h == 0:
        return "TX: trapezoid dP=dN" if abs(dp - dn) < 1e-6 else "O: asym. trapezoid"
    return "O: trapezoid + DC bias"


def build_table(material: str, refeature: bool = False) -> pd.DataFrame:
    scalar = load_scalar()
    cache = DERIVED_ROOT / f"{material.lower()}_features.pkl"
    if cache.exists() and not refeature:
        F = pd.read_pickle(cache)
    else:
        F = extract_material(material, scalar)
        cache.parent.mkdir(parents=True, exist_ok=True)
        F.to_pickle(cache)
    A = scalar.set_index("measurement_id").loc[F.measurement_id].reset_index().join(
        F.drop(columns="measurement_id"))
    A = A[A.quality_accepted & A.duty_quality_accepted & (A.volumetric_loss_w_per_m3 > 0)].copy()
    A["T"] = A.temperature_c.round().astype(int)
    A["Hdc"] = A.h_dc_command_a_per_m.round().astype(int)
    A["dPc"] = A.duty_positive_command.round(1)
    A["dNc"] = A.duty_negative_command.round(1)
    A["combo"] = np.where(A.excitation_type == "trapezoidal",
                          A.dPc.astype(str) + "/" + A.dNc.astype(str), "")
    A["regime"] = [regime(*v) for v in zip(A.excitation_type, A.dPc, A.dNc, A.Hdc)]
    return A.reset_index(drop=True)


def fit_all(A: pd.DataFrame, seed: int = 0):
    D = clm.CoreLossData(A)
    et = A.excitation_type.values; Tv = A["T"].values; Hv = A.Hdc.values; H0 = Hv == 0
    sine = et == "sinusoidal"; trap = et == "trapezoidal"
    tri05 = (et == "triangular") & (np.abs(A.duty_positive_command.values - 0.5) < 1e-6)
    pred = {k: np.full(len(A), np.nan) for k in MODELS}
    params = {}
    combos = sorted(A.loc[trap & H0, "combo"].unique())
    fold = {c: i % 2 for i, c in enumerate(np.random.default_rng(seed).permutation(combos))}
    fA = np.array([fold.get(c, -1) for c in A.combo.values])

    for T in sorted(np.unique(Tv)):
        mT = Tv == T
        ps = clm.fit_steinmetz(clm.P_SE, D, mT & sine & H0)
        pt = clm.fit_steinmetz(clm.P_iGSE, D, mT & tri05 & H0)
        params[(T, 0, "sine")] = ps; params[(T, 0, "tri")] = pt
        pred["SE (sine-fit)"][mT] = clm.P_SE(ps, D, mT)
        pred["MSE (sine-fit)"][mT] = clm.P_MSE(ps, D, mT)
        pred["iGSE (sine-fit)"][mT] = clm.P_iGSE(ps, D, mT)
        pred["iGSE (tri-fit)"][mT] = clm.P_iGSE(pt, D, mT)

        # i2GSE transition term: fold params for held-out trapezoid combos, all-combo params otherwise
        qs = []
        for k in (0, 1):
            mtr = mT & trap & H0 & (fA == k)
            qs.append(clm.fit_transition(D, mtr, clm.P_iGSE(pt, D, mtr)).x)
            params[(T, "fold%d" % k, "transition")] = qs[-1]
        mall = mT & trap & H0
        qa = clm.fit_transition(D, mall, clm.P_iGSE(pt, D, mall)).x
        params[(T, "all", "transition")] = qa

        def assign(out, base_params, rows):
            nt = rows & ~trap
            out[nt] = clm.P_iGSE(base_params, D, nt) + clm.P_transition(qa, D, nt)
            for k in (0, 1):
                hk = rows & trap & (fA == 1 - k)
                if hk.any():
                    out[hk] = clm.P_iGSE(base_params, D, hk) + clm.P_transition(qs[k], D, hk)

        assign(pred["i2GSE (tri-fit)"], pt, mT)

        for h in sorted(np.unique(Hv[mT])):
            mh = mT & (Hv == h)
            if (mh & sine).sum() >= 10:
                psh = clm.fit_steinmetz(clm.P_iGSE, D, mh & sine, ps)
                params[(T, h, "sine_SPG")] = psh
                pred["iGSE+SPG (sine-fit per Hdc)"][mh] = clm.P_iGSE(psh, D, mh)
            if (mh & tri05).sum() >= 10:
                pth = clm.fit_steinmetz(clm.P_iGSE, D, mh & tri05, pt)
                params[(T, h, "tri_SPG")] = pth
                assign(pred["i2GSE+SPG (tri-fit per Hdc)"], pth, mh)
            if (mh & tri05).sum() >= 15:
                lm = clm.fit_loss_map(D.f[mh & tri05], D.Bpk[mh & tri05], D.P[mh & tri05])
                params[(T, h, "loss_map")] = lm
                pred["Composite loss-map (tri D=0.5 per Hdc)"][mh] = clm.P_composite(lm, D, mh)
        print(f"T={T} °C  sine k,a,b={np.exp(ps[0]):.4g},{ps[1]:.3f},{ps[2]:.3f}  "
              f"tri k,a,b={np.exp(pt[0]):.4g},{pt[1]:.3f},{pt[2]:.3f}")
    return D, pred, params


def error_table(A, D, pred):
    keep = A.Hdc.values <= HDC_MAX
    rows = []
    for nm in MODELS:
        e = pred[nm] / D.P - 1
        for rg in REGIMES:
            ee = e[keep & (A.regime.values == rg)]
            rows.append(dict(model=nm, regime=rg, n=len(ee), in_sample=rg in IN_SAMPLE[nm],
                             med_abs_err_pct=np.median(np.abs(ee)) * 100,
                             p95_abs_err_pct=np.percentile(np.abs(ee), 95) * 100,
                             median_bias_pct=np.median(ee) * 100))
    return pd.DataFrame(rows)


def params_table(params):
    rows = []
    for (T, h, kind), v in params.items():
        if kind in ("sine", "tri", "sine_SPG", "tri_SPG") and (not isinstance(h, int) or h <= HDC_MAX):
            rows.append(dict(T_C=T, Hdc_A_per_m=h, fit=kind, k=np.exp(v[0]), alpha=v[1], beta=v[2]))
        elif kind == "transition":
            rows.append(dict(T_C=T, Hdc_A_per_m=0, fit=f"i2GSE term ({h})", kr=np.exp(v[0]),
                             alpha_r=v[1], beta_r=v[2], tau_s=np.exp(v[3]), q_r=v[4]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="N87")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--refeature", action="store_true", help="recompute waveform features")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A = build_table(a.material, a.refeature)
    print(f"{len(A)} accepted measurements")
    D, pred, params = fit_all(A)
    R = error_table(A, D, pred)
    tag = a.material.lower()
    R.to_csv(out / f"{tag}_core_loss_model_errors.csv", index=False)
    params_table(params).to_csv(out / f"{tag}_fitted_parameters.csv", index=False)
    print(R.pivot(index="model", columns="regime", values="med_abs_err_pct")
          .loc[MODELS, REGIMES].round(1).to_string())
    if not a.no_figures:
        import plots
        plots.heatmap(R, out / f"{tag}_fig1_error_heatmap.png")
        plots.mechanisms(A, D, pred, out / f"{tag}_fig2_error_mechanisms.png")


if __name__ == "__main__":
    main()
