# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code for computational physics modeling of **twisted black phosphorus (BP)** and **monolayer hexagonal boron nitride (hBN)**. The focus is electronic band structure and optical properties using tight-binding (TB) models, k·p Hamiltonians, and Wannier interpolation.

## Running Scripts

No build system. Run scripts directly:

```bash
python BP/TB.py          # Twisted bilayer BP (moiré k·p)
python BP/fit_ML.py      # Monolayer BP k·p model
python hBN/hBN.py        # hBN TB/Wannier band comparison
python hBN/hBN_optical.py  # hBN optical properties (production)
jupyter notebook hBN/plot.ipynb
```

Dependencies: `numpy`, `matplotlib`, `scipy`, `jupyter`

## Architecture

### BP/ — Black Phosphorus

- **TB.py**: `TwistedBPModel` — moiré k·p Hamiltonian for 90° twisted bilayer BP. 4-band model per G-vector (Top/Bottom × Conduction/Valence). Implements band structure (folded/unfolded), optical absorption (Kubo-Greenwood), and shift current (SOS method). Complex phase in interlayer coupling (line ~106) is critical for Berry curvature and shift current.
- **fit_ML.py**: `MonolayerBPModel` — simpler 2-band anisotropic k·p model with the same optical suite.
- **\*.dat**: VASP DFT reference data for validation (341.dat, 571.dat, ML.dat).

### hBN/ — Hexagonal Boron Nitride

- **hBN.py**: Standalone functions — `wannier_model()` reads `pwscf_hr.dat`, `TB_model()` builds a 6-shell TB model, `cal_hBN_bands()` overlays TB/Wannier/DFT bands.
- **hBN_optical.py**: Production classes `hBNModel` and `WannierHRModel` with precomputed hopping derivatives for efficient H/v/w evaluation. Includes symmetry verification (D₃ₕ: σʸʸʸ = −σʸˣˣ).
- **Data files**: `pwscf_hr.dat` (Wannier90 HR Hamiltonian), `pwscf_band.dat`, `bands.dat.gnu` — must exist relative to the script folder.

## Key Conventions

**Vectorized API contract** — all models expose:
```python
model.get_hamiltonians(k_points)                    # k_points: (Nk, 2) → H: (Nk, dim_H, dim_H)
model.get_velocity_matrices(k_points)               # → vx, vy each (Nk, dim_H, dim_H)
model.get_generalized_derivative_matrices(k_points) # → w_xx, w_yy, w_xy
```
Diagonalize with `numpy.linalg.eigh` on stacked Hamiltonians. Avoid per-k loops; use NumPy broadcasting.

**Physics parameters**: Keep defaults intact unless explicitly requested — values are literature-fitted. The `test_wannier` flag in `hBN_optical.py` switches between TB and Wannier models.

**Output**: Figures are saved as `.png` files (e.g., `ML_Bands.png`, `wannier_bands.png`) rather than shown interactively.
