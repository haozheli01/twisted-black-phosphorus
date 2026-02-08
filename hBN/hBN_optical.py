'''
Monolayer hBN tight-binding model and optical properties.
Build with the help of Claude Opus 4.6
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 16,
    "mathtext.fontset":'stix',
    "font.serif": ['Arial'], 
}
rcParams.update(config)

##################################################
# Lattice definitions of monolayer hBN (in angstroms)
##################################################
a_1 = np.array([ 2.505754, 0.000000,  0.000000]) 
a_2 = np.array([-1.252877, 2.170046,  0.000000])
a_3 = np.array([ 0.000000, 0.000000, 15.034521])

b_1 = 2 * np.pi * np.cross(a_2, a_3) / np.dot(a_1, np.cross(a_2, a_3))
b_2 = 2 * np.pi * np.cross(a_3, a_1) / np.dot(a_1, np.cross(a_2, a_3))

# 2D components of reciprocal vectors
b1_2d = b_1[:2]
b2_2d = b_2[:2]

# High symmetry points (2D Cartesian)
Gamma = np.array([0.0, 0.0])
K = (b1_2d + b2_2d) / 3.0
M = b1_2d / 2.0

# Unit cell area (in angstroms)
A_uc = abs(a_1[0] * a_2[1] - a_1[1] * a_2[0])
# Effective thickness for monolayer hBN (in angstroms)
d_eff = 1.00
# Volume of the unit cell (in angstroms)
V_uc = A_uc * d_eff
##################################################


def generate_bz_kgrid(n_k):
    """
    Uniform Monkhorst-Pack k-grid covering one BZ (parallelogram cell).
    k = k1*b1 + k2*b2, with k1,k2 = [-0.5, 0.5).

    Parameters:  n_k: grid density per direction (total n_k*n_k points)
    Returns:     k_points (n_k*n_k, 2) in Cartesian coordinates
    """
    frac = np.arange(n_k, dtype=float) / n_k - 0.5 # [-0.5, 0.5)
    # frac = np.arange(n_k, dtype=float) / n_k * 2.0 - 1.0 # [-1, 1)
    K1, K2 = np.meshgrid(frac, frac)
    kx = K1.flatten() * b1_2d[0] + K2.flatten() * b2_2d[0]
    ky = K1.flatten() * b1_2d[1] + K2.flatten() * b2_2d[1]
    return np.column_stack([kx, ky])


class hBNModel:
    """
    6-shell tight-binding model for monolayer hBN (2 orbitals: B, N).

    H_{ab}(k) = E_a * delta_{ab} + sum_R t_{ab}(R) exp(ik*R)

    Hopping parameters fitted to match Wannier90 / DFT band structure.
    All velocity and curvature matrices are computed analytically from
    the hopping vectors.
    """

    def __init__(self,
                 on_site_B=1.042, on_site_N=-3.008, # in eV
                 t_params=None):
        # Basis positions in Fractional coordinates
        basis_frac = {
            "B": np.array([1.0/3.0, 2.0/3.0, 0.5]),
            "N": np.array([2.0/3.0, 1.0/3.0, 0.5])
        }
        # Basis positions in Cartesian coordinates
        self.basis = {
            atom: frac[0]*a_1 + frac[1]*a_2 + frac[2]*a_3
            for atom, frac in basis_frac.items()
        }
        self.on_site = np.array([on_site_B, on_site_N])  # B=0, N=1

        if t_params is None:
            t_params = {
                1: {("B", "N"): -2.72, ("N", "B"): -2.72}, # in eV
                2: {("B", "B"):  0.05, ("N", "N"):  0.24},
                3: {("B", "N"): -0.25, ("N", "B"): -0.25},
                4: {("B", "N"):  0.06, ("N", "B"):  0.06},
                5: {("B", "B"):  0.04, ("N", "N"):  0.03},
                6: {("B", "B"): -0.03, ("N", "N"): -0.02},
            }

        self.dim_H = 2
        max_shell = max(t_params.keys())
        neighbors = self._build_neighbors(max_shell)
        self._precompute_hopping(neighbors, t_params)

        n_hop = sum(len(d['t']) for d in self._hop_data.values())
        print(f"hBN Model initialized: dim_H={self.dim_H}, total_hoppings={n_hop}")

    # ---- Internal construction ----

    def _build_neighbors(self, max_shell, R_range=8, tol=1e-4):
        """Find neighbor shells up to max_shell."""
        pairs = [("B", "B"), ("N", "N"), ("B", "N"), ("N", "B")]
        candidates = []

        for i in range(-R_range, R_range + 1):
            for j in range(-R_range, R_range + 1):
                R = i * a_1 + j * a_2
                for a, b in pairs:
                    d = R + self.basis[b] - self.basis[a]
                    dist = np.linalg.norm(d)
                    if dist < tol:
                        continue
                    candidates.append((dist, d, (a, b)))

        all_dists = sorted(set(round(c[0], 6) for c in candidates))
        shell_dists = []
        for dist in all_dists:
            if not shell_dists or abs(dist - shell_dists[-1]) > tol:
                shell_dists.append(dist)
            if len(shell_dists) >= max_shell:
                break

        neighbors = {s: {p: [] for p in pairs} for s in range(1, max_shell + 1)}
        for dist, d, pair in candidates:
            for shell_idx, shell_dist in enumerate(shell_dists, start=1):
                if abs(dist - shell_dist) <= tol:
                    neighbors[shell_idx][pair].append(d)
                    break
        return neighbors

    def _precompute_hopping(self, neighbors, t_params):
        """
        Group hoppings by matrix element (i,j) and precompute derivative
        weight arrays for efficient vectorized evaluation.
        """
        idx_map = {"B": 0, "N": 1}
        hop_collect = {(0,0): [], (0,1): [], (1,0): [], (1,1): []}

        for shell_idx, pair_dict in neighbors.items():
            if shell_idx not in t_params:
                continue
            for (a, b), vecs in pair_dict.items():
                if (a, b) not in t_params[shell_idx] or len(vecs) == 0:
                    continue
                t_val = t_params[shell_idx][(a, b)]
                ii, jj = idx_map[a], idx_map[b]
                for R in vecs:
                    hop_collect[(ii, jj)].append((t_val, R[:2]))  # 2D only

        self._hop_data = {}
        for ij, data in hop_collect.items():
            if data:
                t_arr = np.array([d[0] for d in data])
                R_arr = np.array([d[1] for d in data])
                Rx, Ry = R_arr[:, 0], R_arr[:, 1]
                self._hop_data[ij] = {
                    't':     t_arr,          # (N_hop,)
                    'R':     R_arr,          # (N_hop, 2)
                    'tRx':   t_arr * Rx,     # for dH/dkx
                    'tRy':   t_arr * Ry,     # for dH/dky
                    'tRxRx': t_arr * Rx**2,  # for d^2H/dkx^2
                    'tRyRy': t_arr * Ry**2,  # for d^2H/dky^2
                    'tRxRy': t_arr * Rx*Ry,  # for d^2H/dkxdky
                }

    # ---- Public API ----

    def get_hamiltonians(self, k_points):
        """
        H_{ij}(k) = E_i * delta_{ij} + sum_R t_{ij}(R) exp(ik * R)

        Parameters:  k_points (Nk, 2)
        Returns:     H (Nk, 2, 2) complex
        """
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        H = np.zeros((Nk, 2, 2), dtype=np.complex128)
        H[:, 0, 0] = self.on_site[0]
        H[:, 1, 1] = self.on_site[1]
        for (i, j), data in self._hop_data.items():
            phases = np.exp(1j * (k_points @ data['R'].T))  # (Nk, N_hop)
            H[:, i, j] += phases @ data['t']
        return H

    def get_velocity_matrices(self, k_points):
        """
        Velocity matrices (dH/dk):
            v_{x,ij}(k) = i sum_R R_x t_{ij}(R) exp(ik * R)
            v_{y,ij}(k) = i sum_R R_y t_{ij}(R) exp(ik * R)

        Returns: vx, vy — each (Nk, 2, 2) complex
        """
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        vx = np.zeros((Nk, 2, 2), dtype=np.complex128)
        vy = np.zeros((Nk, 2, 2), dtype=np.complex128)
        for (i, j), data in self._hop_data.items():
            phases = np.exp(1j * (k_points @ data['R'].T)) # (Nk, N_hop)
            vx[:, i, j] = 1j * (phases @ data['tRx'])
            vy[:, i, j] = 1j * (phases @ data['tRy'])
        return vx, vy

    def get_generalized_derivative_matrices(self, k_points):
        """
        Second derivative (inverse effective mass tensor):
            w_{mu nu,ij}(k) = d^2H_{ij}/(dk_mu dk_nu) = -sum_R [R_mu R_nu t_{ij}(R) exp(ik * R)]

        Returns: w_xx, w_yy, w_xy — each (Nk, 2, 2) complex
        """
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        w_xx = np.zeros((Nk, 2, 2), dtype=np.complex128)
        w_yy = np.zeros((Nk, 2, 2), dtype=np.complex128)
        w_xy = np.zeros((Nk, 2, 2), dtype=np.complex128)
        for (i, j), data in self._hop_data.items():
            phases = np.exp(1j * (k_points @ data['R'].T)) # (Nk, N_hop)
            w_xx[:, i, j] = -(phases @ data['tRxRx'])
            w_yy[:, i, j] = -(phases @ data['tRyRy'])
            w_xy[:, i, j] = -(phases @ data['tRxRy'])
        return w_xx, w_yy, w_xy

    def compute_H_and_velocity(self, k_points):
        """
        Compute H, vx, vy simultaneously — reuses phase factors.
        More efficient than calling get_hamiltonians + get_velocity_matrices.

        Returns: H, vx, vy — each (Nk, 2, 2) complex
        """
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        H  = np.zeros((Nk, 2, 2), dtype=np.complex128)
        vx = np.zeros((Nk, 2, 2), dtype=np.complex128)
        vy = np.zeros((Nk, 2, 2), dtype=np.complex128)
        H[:, 0, 0] = self.on_site[0]
        H[:, 1, 1] = self.on_site[1]
        for (i, j), data in self._hop_data.items():
            phases = np.exp(1j * (k_points @ data['R'].T))  # computed once shape: (Nk, N_hop)
            H[:, i, j]  += phases @ data['t']
            vx[:, i, j]  = 1j * (phases @ data['tRx'])
            vy[:, i, j]  = 1j * (phases @ data['tRy'])
        return H, vx, vy


class WannierHRModel:
    """
    Wannier90 HR model parsed from pwscf_hr.dat.

    Uses the same conventions as hBN.py. All hopping terms are read once
    and stored by (i, j) to allow vectorized evaluation of H, v, w.
    """

    def __init__(self, data_path="pwscf_hr.dat"):
        self.data_path = data_path
        self._read_hr()

        n_hop = sum(len(d['t']) for d in self._hop_data.values())
        print(f"Wannier HR Model initialized: dim_H={self.dim_H}, total_hoppings={n_hop}")

    def _read_hr(self):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()

        num_wann = int(lines[1])
        num_grid = int(lines[2])
        skip_lines = int(3 + np.ceil(num_grid / 15))

        self.dim_H = num_wann
        hop_collect = {}
        for i in range(num_wann):
            for j in range(num_wann):
                hop_collect[(i, j)] = []

        for idx in range(num_wann * num_wann * num_grid):
            line = lines[skip_lines + idx].strip().split()
            Rx, Ry, Rz, m, n = map(int, line[:5])
            H_ij = complex(float(line[5]), float(line[6]))

            m_idx = m - 1
            n_idx = n - 1
            R_vec = Rx * a_1 + Ry * a_2 + Rz * a_3
            hop_collect[(m_idx, n_idx)].append((H_ij, R_vec[:2]))

        self._hop_data = {}
        for ij, data in hop_collect.items():
            if not data:
                continue
            t_arr = np.array([d[0] for d in data], dtype=np.complex128)
            R_arr = np.array([d[1] for d in data])
            Rx, Ry = R_arr[:, 0], R_arr[:, 1]
            self._hop_data[ij] = {
                't':     t_arr,
                'R':     R_arr,
                'tRx':   t_arr * Rx,
                'tRy':   t_arr * Ry,
                'tRxRx': t_arr * Rx**2,
                'tRyRy': t_arr * Ry**2,
                'tRxRy': t_arr * Rx*Ry,
            }

    def get_hamiltonians(self, k_points):
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        H = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        for (i, j), data in self._hop_data.items():
            phases = np.exp(1j * (k_points @ data['R'].T))
            H[:, i, j] += phases @ data['t']
        return H

    def get_velocity_matrices(self, k_points):
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        vx = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        vy = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        for (i, j), data in self._hop_data.items():
            phases = np.exp(1j * (k_points @ data['R'].T))
            vx[:, i, j] = 1j * (phases @ data['tRx'])
            vy[:, i, j] = 1j * (phases @ data['tRy'])
        return vx, vy

    def get_generalized_derivative_matrices(self, k_points):
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        w_xx = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        w_yy = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        w_xy = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        for (i, j), data in self._hop_data.items():
            phases = np.exp(1j * (k_points @ data['R'].T))
            w_xx[:, i, j] = -(phases @ data['tRxRx'])
            w_yy[:, i, j] = -(phases @ data['tRyRy'])
            w_xy[:, i, j] = -(phases @ data['tRxRy'])
        return w_xx, w_yy, w_xy

    def compute_H_and_velocity(self, k_points):
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        H = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        vx = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        vy = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        for (i, j), data in self._hop_data.items():
            phases = np.exp(1j * (k_points @ data['R'].T))
            H[:, i, j] = phases @ data['t']
            vx[:, i, j] = 1j * (phases @ data['tRx'])
            vy[:, i, j] = 1j * (phases @ data['tRy'])
        return H, vx, vy


def resolve_model(model_type="tb", model=None):
    if model is not None:
        return model
    if model_type == "tb":
        return hBNModel()
    if model_type == "wannier":
        return WannierHRModel()
    raise ValueError(f"Unknown model_type: {model_type}")


##################################################
# Band structure
##################################################

def cal_bands(n_seg=200, save_prefix="", model_type="tb", model=None):
    """Plot band structure along Gamma -> M -> K -> Gamma."""
    model = resolve_model(model_type=model_type, model=model)

    path_GM = np.linspace(Gamma, M, n_seg, endpoint=False)
    path_MK = np.linspace(M, K, n_seg, endpoint=False)
    path_KG = np.linspace(K, Gamma, n_seg + 1)
    k_path = np.vstack([path_GM, path_MK, path_KG])

    dists = np.linalg.norm(np.diff(k_path, axis=0), axis=1)
    k_dist = np.concatenate([[0], np.cumsum(dists)])
    sym_pos = [0, k_dist[n_seg], k_dist[2*n_seg], k_dist[-1]]
    sym_labels = [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']

    H_stack = model.get_hamiltonians(k_path)
    energies = np.linalg.eigvalsh(H_stack)

    plt.figure(figsize=(6, 7))
    plt.plot(k_dist, energies[:, 0], 'b-', lw=2.5, label=f"{model_type} model")
    plt.plot(k_dist, energies[:, 1:], 'b-', lw=2.5)

    # Optional: overlay DFT data
    try:
        dft_data = np.loadtxt('bands.dat.gnu')
        dft_kpath = dft_data[:, 0] / dft_data[:, 0].max() * k_dist[-1]
        plt.scatter(dft_kpath, dft_data[:, 1], c='r', s=4, label='DFT', zorder=2)
    except (FileNotFoundError, OSError):
        pass

    for pos in sym_pos:
        plt.axvline(pos, c='gray', ls='-', lw=0.5)
    plt.axhline(0, c='gray', ls='--', lw=0.5)
    plt.xticks(sym_pos, sym_labels)
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(k_dist[0], k_dist[-1])
    plt.ylim(-10, 8)
    plt.tight_layout()
    plt.savefig(f"hBN_bands{save_prefix}.png", dpi=300)
    plt.close()
    print(f"Saved: hBN_bands{save_prefix}.png")
    return model


##################################################
# 3D band structure
##################################################

def plot_3d_bands(n_grid=80, view_elev=25, view_azim=45, save_prefix="", model_type="tb", model=None):
    """Plot 3D band surface on a 2D k-grid covering the BZ."""
    model = resolve_model(model_type=model_type, model=model)
    k_points = generate_bz_kgrid(n_grid)

    H_stack = model.get_hamiltonians(k_points)
    evals = np.linalg.eigvalsh(H_stack)

    kx = k_points[:, 0].reshape(n_grid, n_grid)
    ky = k_points[:, 1].reshape(n_grid, n_grid)
    E_grid = evals.reshape(n_grid, n_grid, evals.shape[1])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(kx, ky, E_grid[:, :, 0], cmap='viridis', alpha=0.9,
                    edgecolor='none', antialiased=False)
    if E_grid.shape[2] > 1:
        ax.plot_surface(kx, ky, E_grid[:, :, 1], cmap='plasma', alpha=0.9,
                        edgecolor='none', antialiased=False)

    ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    ax.set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    ax.set_zlabel('Energy (eV)')
    ax.set_title('hBN 3D Band Structure')
    ax.set_box_aspect((1, 1, 1.5))
    ax.view_init(elev=view_elev, azim=view_azim)

    plt.tight_layout()
    plt.savefig(f"hBN_3D{save_prefix}.png", dpi=300)
    plt.close()
    print(f"Saved: hBN_3D{save_prefix}.png")


##################################################
# Optical absorption
##################################################

def calculate_optical_conductivity(E_range=(3.0, 10.0), n_E=500, eta=0.050,
                                   n_k=120, save_prefix="", model_type="tb", model=None):
    r"""
    Interband optical absorption via Kubo-Greenwood formula.

        epsilon^{ii}(omega) \propto sum_{vc} \int_{BZ} d^2k  |⟨c|v_i|v⟩|^2 / \Delta E^2  \cdot \delta(\Delta E - \omega)

    The plotted quantity is \omega \cdot \epsilon_2(\omega) \propto absorption coefficient.
    delta-function approximated by Lorentzian with width eta.
    """
    print(f"Calculating optical absorption (n_k={n_k}, eta={eta*1000:.0f} meV)...")
    model = resolve_model(model_type=model_type, model=model)
    k_points = generate_bz_kgrid(n_k)
    Nk = len(k_points)

    # Diagonalize + velocity (shared phase computation)
    H_stack, vx_orb, vy_orb = model.compute_H_and_velocity(k_points)
    evals, evecs = np.linalg.eigh(H_stack)

    # Velocity in eigenbasis
    U_dag = np.conj(np.transpose(evecs, (0, 2, 1)))
    vx_eig = U_dag @ vx_orb @ evecs
    vy_eig = U_dag @ vy_orb @ evecs

    Mx2 = np.abs(vx_eig)**2  # (Nk, 2, 2)
    My2 = np.abs(vy_eig)**2

    Nb = evals.shape[1]
    mid = Nb // 2  # = 1 for 2-band hBN
    omegas = np.linspace(E_range[0], E_range[1], n_E)

    sigma_xx = np.zeros(n_E)
    sigma_yy = np.zeros(n_E)

    for i in range(mid):
        for j in range(mid, Nb):
            delta_E = evals[:, j] - evals[:, i]  # (Nk,)
            M_x = Mx2[:, i, j] / delta_E**2
            M_y = My2[:, i, j] / delta_E**2

            diff = omegas[:, None] - delta_E[None, :]  # (n_E, Nk)
            lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)

            sigma_xx += np.sum(M_x[None, :] * lorentz, axis=1)
            sigma_yy += np.sum(M_y[None, :] * lorentz, axis=1)

    sigma_xx /= Nk
    sigma_yy /= Nk

    absorption_xx = sigma_xx * omegas
    absorption_yy = sigma_yy * omegas

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(omegas, absorption_xx, 'r-', label='x-polarized', lw=2)
    plt.plot(omegas, absorption_yy, 'b--', label='y-polarized', lw=2)
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Optical Absorption (a.u.)')
    plt.title(fr'hBN Optical Absorption ($\eta$={eta*1000:.0f} meV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)
    plt.savefig(f"hBN_absorption{save_prefix}.png", dpi=300)
    plt.close()
    print(f"Saved: hBN_absorption{save_prefix}.png")

    return omegas, absorption_xx, absorption_yy


##################################################
# Transition matrix element map
##################################################

def plot_transition_matrix_elements(band_indices=None, n_k=120, save_prefix="", model_type="tb", model=None):
    """
    Plot |<j|v_mu|i>|^2 in k-space as contour maps.

    Parameters:
        band_indices : (i, j) band pair (0-based). Default: VBM -> CBM.
    """
    model = resolve_model(model_type=model_type, model=model)
    k_points = generate_bz_kgrid(n_k)

    if band_indices is None:
        band_i, band_j = 0, 1
    else:
        band_i, band_j = band_indices

    print(f"Mapping transition matrix elements: Band {band_i} -> Band {band_j}")

    H_stack, vx_orb, vy_orb = model.compute_H_and_velocity(k_points)
    _, evecs = np.linalg.eigh(H_stack)

    u_i = evecs[:, :, band_i]  # (Nk, dim_H)
    u_j = evecs[:, :, band_j]

    M_x = np.einsum('ka,kab,kb->k', u_j.conj(), vx_orb, u_i)
    M_y = np.einsum('ka,kab,kb->k', u_j.conj(), vy_orb, u_i)

    Z_x = np.abs(M_x)**2
    Z_y = np.abs(M_y)**2

    kx = k_points[:, 0].reshape(n_k, n_k)
    ky = k_points[:, 1].reshape(n_k, n_k)
    Z_x = Z_x.reshape(n_k, n_k)
    Z_y = Z_y.reshape(n_k, n_k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    vmax = np.percentile(np.concatenate([Z_x.ravel(), Z_y.ravel()]), 99)

    c1 = axes[0].contourf(kx, ky, Z_x, levels=40, cmap='plasma', vmin=0, vmax=vmax)
    axes[0].set_title(r'$|\langle c|v_x|v\rangle|^2$ (x-pol)')
    axes[0].set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    axes[0].set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    axes[0].set_aspect('equal')
    plt.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(kx, ky, Z_y, levels=40, cmap='plasma', vmin=0, vmax=vmax)
    axes[1].set_title(r'$|\langle c|v_y|v\rangle|^2$ (y-pol)')
    axes[1].set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    axes[1].set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    axes[1].set_aspect('equal')
    plt.colorbar(c2, ax=axes[1])

    plt.suptitle(f"Transition Matrix Elements: Band {band_i} -> Band {band_j}")
    plt.tight_layout()
    plt.savefig(f"hBN_TME{save_prefix}.png", dpi=300)
    plt.close()
    print(f"Saved: hBN_TME{save_prefix}.png")


##################################################
# Shift current (SOS method)
##################################################

def calculate_shift_current(E_range=(3.0, 10.0), n_E=400, eta=0.050,
                            n_k=120, comp=('y', 'y', 'y'),
                            band_window=None, save_prefix="",
                            model_type="tb", model=None):
    r"""
    Shift current sigma^{abc}(ω) via gauge-invariant Sum-Over-States method.

    Ref: Phys. Rev. B 61, 5337 (2000)

        \sigma^{abc}(\omega) = C * \sum_{nm}[ f_{nm} \Im[r^b_{mn} (r^c_{nm})_{;a} + r^c_{mn} (r^b_{nm})_{;a}] * \delta(\omega - \omega_{mn})]

    where:
        C = (\pi e^3) / (\hbar^2 V_{uc})
        r^b_{mn} = v^b_{mn} / (i \omega_{mn})
        (r^c_{nm})_{;a} = (-1/i\omega_{nm}) [ term_A/\omega_{nm} + term_B + term_C ]
        term_A = v^c_{nm} \delta^a_{nm} + v^a_{nm} \delta^c_{nm}       (\delta^a_{nm} = v^a_{nn} - v^a_{mm})
        term_B = \sum_{p\neq n,m} [v^c_{np} v^a_{pm}/\omega_{pm} - v^a_{np} v^c_{pm}/\omega_{np}]
        term_C = - v^{ab}_{nm}  (generalized derivative of velocity)   ! This term is often neglected in literature but important for TB models !

    For linearly polarized light, the third-rank response tensor is symmetric under b <--> c. And we consider linearly polarized light with b = c here.

    For 2-band hBN: term_B = 0 (no intermediate states).

    Single-layer hBN (point group D3h) allows the following
    nonzero in-plane components of the rank-3 polar tensor:
        sigma^{yyy} = -sigma^{yxx} = -sigma^{xxy} = -sigma^{xyx}
        All other components are forbidden by symmetry (≈ 0 numerically).

    Parameters:
        comp : (a, b, c) — tensor component directions
        band_window : (v_start, v_end, c_start, c_end) or None
    """
    a_dir, b_dir, c_dir = comp
    print(f"Calculating shift current sigma^{{{a_dir}{b_dir}{c_dir}}}(omega) "
          f"(n_k={n_k}, η={eta*1000:.0f} meV)...")

    model = resolve_model(model_type=model_type, model=model)
    k_points = generate_bz_kgrid(n_k)
    Nk = len(k_points)

    # Diagonalize + velocity (shared phases)
    H_stack, vx_orb, vy_orb = model.compute_H_and_velocity(k_points)
    w_xx, w_yy, w_xy = model.get_generalized_derivative_matrices(k_points)
    evals, evecs = np.linalg.eigh(H_stack)
    Nb = evals.shape[1]

    # Velocity in eigenbasis
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))

    v_map = {
        'x': U_dag @ vx_orb @ U,
        'y': U_dag @ vy_orb @ U
    }
    w_map = {
        'xx': U_dag @ w_xx @ U,
        'yy': U_dag @ w_yy @ U,
        'xy': U_dag @ w_xy @ U,
        'yx': U_dag @ w_xy @ U # symmetry
    }
    v_a, v_b, v_c = v_map[a_dir], v_map[b_dir], v_map[c_dir]
    w_ac_key = a_dir + c_dir
    w_ac = w_map[w_ac_key]

    # Band selection
    mid = Nb // 2
    if band_window is None:
        v_idx = list(range(mid))
        c_idx = list(range(mid, Nb))
    else:
        v_idx = list(range(band_window[0], band_window[1]+1))
        c_idx = list(range(band_window[2], band_window[3]+1))

    omegas = np.linspace(E_range[0], E_range[1], n_E)
    sigma = np.zeros(n_E)

    print(f"  Transitions: {len(v_idx)} val x {len(c_idx)} cond, Nk={Nk}")

    for n in v_idx:
        for m in c_idx:
            w_mn = evals[:, m] - evals[:, n]  # (Nk,), > 0 for insulator
            nonzero = w_mn > 1e-5

            f_n = 1.0 if n < mid else 0.0
            f_m = 1.0 if m < mid else 0.0
            f_nm = f_n - f_m
            if f_nm == 0.0:
                continue

            # --- r^b_mn = v^b_{mn} / (i ω_mn) ---
            r_b_mn = np.zeros(Nk, dtype=np.complex128)
            r_b_mn[nonzero] = v_b[nonzero, m, n] / (1j * w_mn[nonzero])

            # --- Term A: intraband velocity differences ---
            termA = np.zeros(Nk, dtype=np.complex128)
            delta_a = v_a[nonzero, n, n] - v_a[nonzero, m, m]
            delta_c = v_c[nonzero, n, n] - v_c[nonzero, m, m]
            termA[nonzero] = (v_c[nonzero, n, m] * delta_a +
                              v_a[nonzero, n, m] * delta_c)
            termA[nonzero] /= (-w_mn[nonzero])

            # --- Term B: sum over intermediate states ---
            termB = np.zeros(Nk, dtype=np.complex128)
            for p in range(Nb):
                if p == n or p == m:
                    continue
                w_np = evals[:, n] - evals[:, p]
                w_pm = evals[:, p] - evals[:, m]
                p_mask = nonzero & (np.abs(w_np) > 1e-5) & (np.abs(w_pm) > 1e-5)
                if np.any(p_mask):
                    termB[p_mask] += (
                        v_c[p_mask, n, p] * v_a[p_mask, p, m] / w_pm[p_mask]
                      - v_a[p_mask, n, p] * v_c[p_mask, p, m] / w_np[p_mask]
                    )
            
            termC = -w_ac[:, n, m]
            # --- Generalized derivative ---
            # K_nm = termA + termB
            K_nm = termA + termB + termC # important!
            r_deriv = np.zeros(Nk, dtype=np.complex128)
            r_deriv[nonzero] = K_nm[nonzero] / (-1j * (-w_mn[nonzero]))

            # --- Shift current weight ---
            weight = f_nm * np.imag(r_b_mn * r_deriv)

            # --- Lorentzian broadening ---
            diff = omegas[:, None] - w_mn[None, :]       # (n_E, Nk)
            lorentz = (1.0/np.pi) * eta / (diff**2 + eta**2)
            sigma += np.sum(lorentz * weight[None, :], axis=1)

    sigma /= Nk

    # Constant prefactor conversion
    e_charge = 1.602176634e-19  # C
    hbar = 1.054571817e-34      # J·s
    prefactor = (2 * np.pi * e_charge**2) / (hbar * A_uc) * 1E6  # another factor of 1/hbar give to the delta function
    sigma *= prefactor # in ( \muA * angstrom)/V^2

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(omegas, sigma, 'r-', lw=2,
             label=fr'$\sigma^{{{a_dir}{b_dir}{c_dir}}}(\omega)$')
    plt.axhline(0, color='k', lw=0.5, ls='--')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Shift Conductivity (μA / (V²·Å))')
    plt.title('hBN Shift Current Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)

    suffix = f"_{a_dir}{b_dir}{c_dir}{save_prefix}"
    plt.savefig(f"hBN_shift{suffix}.png", dpi=300)
    plt.tight_layout()
    plt.close()
    print(f"Saved: hBN_shift{suffix}.png")

    return omegas, sigma


##################################################
# Main
##################################################

if __name__ == "__main__":

    # # 1. Band structure
    # print("=" * 60)
    # print("[1] Band Structure")
    # print("=" * 60)
    # cal_bands()

    # # 2. 3D bands
    # print("\n" + "=" * 60)
    # print("[2] 3D Band Structure")
    # print("=" * 60)
    # plot_3d_bands(n_grid=80)

    # # 3. Optical absorption
    # print("\n" + "=" * 60)
    # print("[3] Optical Absorption")
    # print("=" * 60)
    # calculate_optical_conductivity(n_k=1000, eta=0.01, E_range=(2.0, 8.0))

    # # 4. Transition matrix elements
    # print("\n" + "=" * 60)
    # print("[4] Transition Matrix Elements")
    # print("=" * 60)
    # plot_transition_matrix_elements(n_k=1000)

    # # 5. Shift current — symmetry check
    # print("\n" + "=" * 60)
    # print("[5] Shift Current — Symmetry Check")
    # print("=" * 60)

    # sc_kwargs = dict(n_k=500, eta=0.02, E_range=(2.0, 8.0))

    # omegas, sigma_yyy = calculate_shift_current(comp=('y','y','y'), **sc_kwargs)
    # _,      sigma_yxx = calculate_shift_current(comp=('y','x','x'), **sc_kwargs)
    # _,      sigma_xxx = calculate_shift_current(comp=('x','x','x'), **sc_kwargs)
    # _,      sigma_xyy = calculate_shift_current(comp=('x','y','y'), **sc_kwargs)

    # # Comparison plot
    # plt.figure(figsize=(8, 6))
    # plt.plot(omegas, sigma_yyy, 'r-',  lw=2, label=r'$\sigma^{yyy}$')
    # plt.plot(omegas, sigma_yxx, 'b--', lw=2, label=r'$\sigma^{yxx}$')
    # plt.plot(omegas, sigma_xxx, 'g:',  lw=2, label=r'$\sigma^{xxx}$')
    # plt.plot(omegas, sigma_xyy, 'y:',  lw=2, label=r'$\sigma^{xyy}$')
    # plt.axhline(0, color='k', lw=0.5, ls='--')
    # plt.xlabel('Photon Energy (eV)')
    # plt.ylabel('Shift Conductivity (μA / (V²·Å))')
    # plt.title(r'hBN Shift Current')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.xlim((2.0, 8.0))
    # plt.tight_layout()
    # plt.savefig("hBN_shift_comparison.png", dpi=300)
    # plt.close()

    # # Verification
    # print("\n" + "=" * 60)
    # print("Symmetry Verification:")
    # print(f"  |sigma^yyy| peak: {np.max(np.abs(sigma_yyy)):.6f}")
    # print(f"  |sigma^yxx| peak: {np.max(np.abs(sigma_yxx)):.6f}")
    # print(f"  |sigma^xxx| max:  {np.max(np.abs(sigma_xxx)):.6f}  (should be ≈ 0)")
    # print(f"  |sigma^xyy| max:  {np.max(np.abs(sigma_xyy)):.6f}  (should be ≈ 0)")
    # mask = np.abs(sigma_yxx) > 1e-8
    # if np.any(mask):
    #     ratio = np.mean(sigma_yyy[mask] / (-sigma_yxx[mask]))
    #     print(f"  sigma^yyy / (-sigma^yxx): {ratio:.4f}  (should be ≈ 1)")
    # print("=" * 60)

    # 6. Wannier HR model examples
    print("\n" + "=" * 60)
    print("[6] Wannier HR Model Examples")
    print("=" * 60)

    cal_bands(model_type="wannier", save_prefix="_wannier")
    calculate_optical_conductivity(model_type="wannier", n_k=500, eta=0.02,
                                   E_range=(2.0, 8.0), save_prefix="_wannier")
    plot_transition_matrix_elements(model_type="wannier", n_k=500, save_prefix="_wannier")

    sc_kwargs_w = dict(n_k=500, eta=0.02, E_range=(2.0, 8.0), save_prefix="_wannier",
                       model_type="wannier")
    calculate_shift_current(comp=('y','y','y'), **sc_kwargs_w)
    omegas, sigma_yyy = calculate_shift_current(comp=('y','y','y'), **sc_kwargs_w)
    _,      sigma_yxx = calculate_shift_current(comp=('y','x','x'), **sc_kwargs_w)
    _,      sigma_xxx = calculate_shift_current(comp=('x','x','x'), **sc_kwargs_w)
    _,      sigma_xyy = calculate_shift_current(comp=('x','y','y'), **sc_kwargs_w)

    # Comparison plot
    plt.figure(figsize=(8, 6))
    plt.plot(omegas, sigma_yyy, 'r-',  lw=2, label=r'$\sigma^{yyy}$')
    plt.plot(omegas, sigma_yxx, 'b--', lw=2, label=r'$\sigma^{yxx}$')
    plt.plot(omegas, sigma_xxx, 'g:',  lw=2, label=r'$\sigma^{xxx}$')
    plt.plot(omegas, sigma_xyy, 'y:',  lw=2, label=r'$\sigma^{xyy}$')
    plt.axhline(0, color='k', lw=0.5, ls='--')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Shift Conductivity (μA / (V²·Å))')
    plt.title(r'hBN Shift Current')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim((2.0, 8.0))
    plt.tight_layout()
    plt.savefig("hBN_shift_comparison_wannier.png", dpi=300)
    plt.close()

    # Verification
    print("\n" + "=" * 60)
    print("Symmetry Verification:")
    print(f"  |sigma^yyy| peak: {np.max(np.abs(sigma_yyy)):.6f}")
    print(f"  |sigma^yxx| peak: {np.max(np.abs(sigma_yxx)):.6f}")
    print(f"  |sigma^xxx| max:  {np.max(np.abs(sigma_xxx)):.6f}  (should be ≈ 0)")
    print(f"  |sigma^xyy| max:  {np.max(np.abs(sigma_xyy)):.6f}  (should be ≈ 0)")
    mask = np.abs(sigma_yxx) > 1e-8
    if np.any(mask):
        ratio = np.mean(sigma_yyy[mask] / (-sigma_yxx[mask]))
        print(f"  sigma^yyy / (-sigma^yxx): {ratio:.4f}  (should be ≈ 1)")
    print("=" * 60)