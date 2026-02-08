# Copilot instructions for this repo

## Project overview
- Research code for twisted black phosphorus (BP) and monolayer hBN models.
- Two main areas:
  - BP/ contains k·p Hamiltonian models and optical calculations for BP.
  - hBN/ contains tight-binding and Wannier HR models plus optical spectra for hBN.

## Key modules and data flow
- BP/TB.py defines `TwistedBPModel` (moire k·p) and helpers like `cal_bands()` that build k-paths, compute stacked Hamiltonians, then plot/save figures.
- BP/fit_ML.py defines `MonolayerBPModel` and higher-level routines for band plots and optical spectra; workflow is: build k-grid → diagonalize H → compute velocity/curvature → transform to eigenbasis → integrate spectra.
- hBN/hBN.py provides two band models: `wannier_model()` reads pwscf_hr.dat and `TB_model()` builds a 6-shell tight-binding model; `cal_hBN_bands()` compares TB/Wannier/DFT data (bands.dat.gnu).
- hBN/hBN_optical.py expands hBN into reusable model classes (`hBNModel`, `WannierHRModel`) with vectorized H/v/w operators and band/optical routines.

## Conventions and patterns
- Models expose vectorized APIs taking `k_points` shaped (Nk, 2) and return stacked matrices (Nk, dim_H, dim_H). Preserve this shape contract when adding features.
- Use `numpy.linalg.eigvalsh` or `numpy.linalg.eigh` on stacked Hamiltonians for band energies/eigenvectors.
- Plotting uses Matplotlib with saved figures (e.g., ML_Bands.png, ML_Optical.png, wannier_bands.png) rather than interactive display.

## External data dependencies
- hBN workflows expect Wannier/DFT files in hBN/: pwscf_hr.dat, pwscf_band.dat, bands.dat.gnu.
- Some scripts assume these files exist and will raise if missing; keep paths relative to the script folder.

## Running scripts (typical)
- Run individual analysis scripts directly with Python, for example: BP/TB.py, BP/fit_ML.py, hBN/hBN.py, hBN/hBN_optical.py.
- No centralized build/test system detected in this repo.

## When modifying
- Keep physics parameter defaults intact unless requested; many values are literature-fit.
- Maintain vectorized, broadcast-friendly implementations (avoid per-k loops when possible).
- Reference functions/classes where patterns are exemplified: `TwistedBPModel` in BP/TB.py, `MonolayerBPModel` in BP/fit_ML.py, and `hBNModel`/`WannierHRModel` in hBN/hBN_optical.py.