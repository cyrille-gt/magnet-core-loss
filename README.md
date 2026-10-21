# Core-loss model comparison (MagNet)

| File | Role |
|---|---|
| `magnet_io.py` | Download (with sha256 check) and loading of MagNet data in `~\data\magnet\steady-state-training-ready` |
| `waveform_features.py` | Features computed on measured B(t): ΔB, ⟨(dB/dt)²⟩, ⟨\|dB/dt\|^α⟩ (α = 0.8–3.0), piecewise-linear segment slopes |
| `core_loss_models.py` | SE, MSE, iGSE, i²GSE, SPG (per-H_dc refit), composite loss map |
| `compare_n87.py` | Full pipeline: fits, errors per regime, CSV + figures in `results\` |
| `plots.py` | Figures (error heatmap, error vs D / H_dc / dwell) |

Setup
```
pip install -r requirements.txt
set MAGNET_DATA_DIR=~\data\magnet      # optional; this is the default
```
Data are not versioned (see `.gitignore`); `magnet_io.download()` fetches any missing
shard from Hugging Face and checks its sha256 against the release manifest.

Data layout (under MAGNET_DATA_DIR)
```
~\data\magnet\steady-state-training-ready\data\scalar\part-00000.parquet   (all materials)
~\data\magnet\steady-state-training-ready\data\sequence\part-000NN.parquet (N87 = parts 9, 10, 11)
~\data\magnet\derived\n87_features.pkl                                    (feature cache, ~12 min to rebuild)
```

import sys; sys.path.insert(0, r"~\code\python")
import compare_n87
A = compare_n87.build_table("N87")          # add refeature=True to recompute features
D, pred, params = compare_n87.fit_all(A)
R = compare_n87.error_table(A, D, pred)
```

Notes
* pandas.read_parquet is broken in this env (pyarrow 25 / pandas 2.3): use pyarrow directly, cache as pickle.
* MagNet trapezoid dwell intervals are not flat; models are evaluated on the measured B(t).
* i²GSE parameters fitted on MagNet: τ collapses to ≤100 ns, so the term acts as a per-transition
  |dB/dt|^≈2 correction rather than a relaxation.
* Data: Princeton MagNet, CC BY 4.0 — cite Li et al., IEEE TPEL 2023, doi:10.1109/TPEL.2023.3309232.
