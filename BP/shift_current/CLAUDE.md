# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computational physics project for studying **twisted bilayer black phosphorus (BP)** optical and electronic properties. Calculates band structures, optical conductivity, shift current spectra, and excitonic effects (BSE) for twisted BP moiré systems.

## Running

Both scripts are standalone. Run directly with Python:

```bash
conda activate QTrans
python continuum.py
python effective_model.py
```

Rememeber to use the QTrans env to run python!

Computation parameters are configured in the `if __name__ == "__main__"` blocks at the bottom of each file. Toggle calculations by commenting/uncommenting function calls there.

Dependencies: `numpy`, `matplotlib` (with Arial font). No build system or tests.

## Architecture

Two parallel implementations of the same physics with a shared API pattern:

### `continuum.py` — Continuum (k·p) model
- **`TwistedBPModel`**: Plane-wave expansion in moiré G-vector basis. Layers are related by 90° rotation hardcoded into the Hamiltonian construction.
- Parameters: `N_shell` (G-vector shells, controls basis size → `dim_H = 4 * dim_G`), `E_field` (perpendicular electric field).
- G-vectors filtered by circular cutoff (`G_norm <= N_shell * delta_g`).
- k-independent parts (interlayer coupling, second derivatives) are precomputed in `__init__` for performance.
- Hamiltonian is 4×4 block structure per G-point: `[Tc, Tv, Bc, Bv]` (Top/Bottom × Conduction/Valence), indexed as `r*4+0..3`.
- Reference data: `571.dat` contains VASP band structure for validation of folded bands.

### `effective_model.py` — Tight-binding effective model
- **`TwistedBPModel`**: Real-space tight-binding with explicit hopping parameters (`t1`–`t10` intralayer, `t1p`–`t4p` interlayer).
- Parameters: `N_top`, `N_bottom` (layer counts), `twist_angle` (arbitrary rotation between layers).
- Interlayer coupling transformed from band basis to sublattice basis via Gamma-point eigenstates (`_precompute_coupling`).
- Hamiltonian is 4×4: two 2×2 sublattice blocks `[[a, z], [z*, a]]` per layer, coupled by constant `V_sub`.
- Supports monolayer mode when `N_bottom=0`.

### Shared API on both `TwistedBPModel` classes
- `get_hamiltonians(k_points)` → `(Nk, dim_H, dim_H)` Hermitian matrices
- `get_velocity_matrices(k_points)` → `vx, vy` each `(Nk, dim_H, dim_H)` — first derivatives `dH/dk`
- `get_generalized_derivative_matrices(k_points)` → `w_xx, w_yy, w_xy` — second derivatives `d²H/dk_μdk_ν`

All accept `k_points` as `(Nk, 2)` arrays and are fully vectorized over k-points.

### Standalone analysis functions (present in both files)
- `cal_bands` — Band structure along X→Γ→Y with zone folding
- `plot_3d_bands` — 3D surface plot of bands on 2D k-grid
- `calculate_optical_conductivity` — Inter-band optical conductivity σ(ω) via Kubo formula
- `calculate_shift_current` / `calculate_z_shift_current` — Bulk photovoltaic shift current (in-plane and out-of-plane z-component)
- `calculate_bse_z_shift_current` — Excitonic shift current via Bethe-Salpeter equation with Keldysh potential
- `plot_transition_matrix_elements` — k-resolved matrix element visualization

## Key Physics Conventions

- Units: energies in eV, lengths in Å, k-vectors in Å⁻¹
- Shift current output in μA·Å/V²
- Moiré reciprocal vector: `delta_g = 2π|1/b - 1/a|` where `a=3.296 Å`, `b=4.588 Å`
- `E_field` couples as `±E·d/2` to top/bottom layers (sign convention: opposite to physical direction in continuum model)
- Eigenvalue sorting assumes half-filling (VBM = max of lower half of bands)
