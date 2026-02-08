import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

##################################################
# Lattice vectors (in angstroms)
##################################################
a_1 = np.array([2.505754,   0.000000,   0.000000])
a_2 = np.array([-1.252877,  2.170046,   0.000000])
a_3 = np.array([0.000000,   0.000000,  15.034521])

# reciprocal lattice vectors
b_1 = 2 * np.pi * np.cross(a_2, a_3) / np.dot(a_1, np.cross(a_2, a_3))
b_2 = 2 * np.pi * np.cross(a_3, a_1) / np.dot(a_1, np.cross(a_2, a_3))
b_3 = 2 * np.pi * np.cross(a_1, a_2) / np.dot(a_1, np.cross(a_2, a_3))

# High symmetry points
Gamma = np.array([0, 0, 0])
K = (b_1 + b_2) / 3
M = b_1 / 2
##################################################
# Models
##################################################
def wannier_model(k_points):

    k_points = np.atleast_2d(k_points)
    num_k = len(k_points)
    kx, ky = k_points[:, 0], k_points[:, 1]

    data_path = 'pwscf_hr.dat'
    with open(data_path, 'r') as f:
        lines = f.readlines()
        num_wann = int(lines[1])
        num_grid = int(lines[2])
        skip_lines = int(3 + np.ceil(num_grid/15))
    # print(f'Number of Wannier functions: {num_wann}')
    # print(f'Number of k-points: {num_grid}')
    # print(f'Number of lines to skip: {skip_lines}')

    HK = np.zeros((num_k, num_wann, num_wann), dtype=complex)
    for i in range(num_wann**2 * num_grid):
        line = lines[skip_lines + i].strip().split()
        Rx, Ry, Rz, m, n = map(int, line[:5])
        H_ij = complex(float(line[5]), float(line[6]))
        m_wann = m - 1
        n_wann = n - 1
        R_vec = Rx * a_1 + Ry * a_2 + Rz * a_3 # (3,)
        phase = np.exp(1j * k_points @ R_vec)  # (Nk,)
        HK[:, m_wann, n_wann] += H_ij * phase

    return HK

def get_neighbors(a_1, a_2, basis, max_shell=6, R_range=8, tol=1e-4):
    """
    Automatically find neighbor shells for a two-atom basis.
    
    Args:
        a_1, a_2: lattice vectors
        basis: dict {"B": cart_pos, "N": cart_pos}
        max_shell: number of shells to find
        R_range: search range for lattice translations
        tol: tolerance for grouping distances
    
    Returns:
        neighbors: dict {shell_idx: {(atom_a, atom_b): [displacement vectors]}}
    """
    pairs = [("B", "B"), ("N", "N"), ("B", "N"), ("N", "B")]
    candidates = []  # list of (distance, displacement_vector, (atom_a, atom_b))

    for i in range(-R_range, R_range + 1):
        for j in range(-R_range, R_range + 1):
            R = i * a_1 + j * a_2
            for a, b in pairs:
                d = R + basis[b] - basis[a]
                dist = np.linalg.norm(d)
                if dist < tol:
                    continue
                candidates.append((dist, d, (a, b)))

    # Find unique shell distances
    all_dists = sorted(set(round(c[0], 6) for c in candidates))
    shell_dists = []
    for dist in all_dists:
        if not shell_dists or abs(dist - shell_dists[-1]) > tol:
            shell_dists.append(dist)
        if len(shell_dists) >= max_shell:
            break

    # Build neighbor dict
    neighbors = {s: {p: [] for p in pairs} for s in range(1, max_shell + 1)}
    for dist, d, pair in candidates:
        for shell_idx, shell_dist in enumerate(shell_dists, start=1):
            if abs(dist - shell_dist) <= tol:
                neighbors[shell_idx][pair].append(d)
                break

    return neighbors

def TB_model(k_points):
    k_points = np.atleast_2d(k_points)
    num_k = len(k_points)

    # Basis positions in Cartesian coordinates (B and N)
    basis_frac = {
        "B": np.array([1.0/3.0, 2.0/3.0, 0.5]),
        "N": np.array([2.0/3.0, 1.0/3.0, 0.5])
    }
    basis = {
        "B": basis_frac["B"][0] * a_1 + basis_frac["B"][1] * a_2 + basis_frac["B"][2] * a_3,
        "N": basis_frac["N"][0] * a_1 + basis_frac["N"][1] * a_2 + basis_frac["N"][2] * a_3
    }

    # On-site energies (eV)
    on_site = {"B": 1.042, "N": -3.008}

    # Hopping parameters: {shell_index: {(atom_a, atom_b): t_value}}
    t_params = {
        1: {("B", "N"): -2.72, ("N", "B"): -2.72},
        2: {("B", "B"):  0.05, ("N", "N"):  0.24},
        3: {("B", "N"): -0.25, ("N", "B"): -0.25},
        4: {("B", "N"):  0.06, ("N", "B"):  0.06},
        5: {("B", "B"):  0.04, ("N", "N"):  0.03},
        6: {("B", "B"): -0.03, ("N", "N"): -0.02},
    }

    max_shell = max(t_params.keys())
    neighbors = get_neighbors(a_1, a_2, basis, max_shell=max_shell)
    # print("Neighbor shells and counts:")
    # for shell_idx, pair_dict in neighbors.items():
    #     counts = {pair: len(vecs) for pair, vecs in pair_dict.items()}
    #     print(f" Shell {shell_idx}: {counts}")

    idx_map = {"B": 0, "N": 1}
    H_k = np.zeros((num_k, 2, 2), dtype=complex)
    H_k[:, 0, 0] = on_site["B"]
    H_k[:, 1, 1] = on_site["N"]

    for shell_idx, pair_dict in neighbors.items():
        if shell_idx not in t_params:
            continue
        for (a, b), vecs in pair_dict.items():
            if (a, b) not in t_params[shell_idx] or len(vecs) == 0:
                continue
            t_val = t_params[shell_idx][(a, b)]
            vecs_arr = np.array(vecs)                          # (n_neighbors, 3)
            phases = np.exp(1j * (k_points @ vecs_arr.T))     # (num_k, n_neighbors)
            H_k[:, idx_map[a], idx_map[b]] += t_val * phases.sum(axis=1)

    return H_k

def cal_hBN_bands():
    """
    Calculate and plot band structure along high-symmetry path.
    """
    
    # High symmetry path: Gamma -> M -> K -> Gamma
    n_seg = 100
    
    path_GM = np.linspace(Gamma, M, n_seg, endpoint=False)
    path_MK = np.linspace(M, K, n_seg, endpoint=False)
    path_KG = np.linspace(K, Gamma, n_seg + 1)
    
    k_path = np.vstack([path_GM, path_MK, path_KG])
    
    # Calculate distances
    dists = np.linalg.norm(np.diff(k_path, axis=0), axis=1)
    k_dist = np.concatenate([[0], np.cumsum(dists)])
    
    # High symmetry positions
    sym_pos = [0, k_dist[n_seg], k_dist[2*n_seg], k_dist[-1]]
    sym_labels = [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']
    
    # Diagonalize
    H_stack_wannier = wannier_model(k_path)
    H_stack_TB = TB_model(k_path)
    energies_wannier = np.linalg.eigvalsh(H_stack_wannier)
    energies_TB = np.linalg.eigvalsh(H_stack_TB)
    
    # Plot
    # wannier_model
    plt.figure(figsize = (5.6,6.4))
    plt.plot(k_dist, energies_wannier[:, 0], 'b-', lw=3.2)
    plt.plot(k_dist, energies_wannier[:, 1], 'b-', lw=3.2, label='wannier_hr')
    # TB_model
    plt.plot(k_dist, energies_TB[:, 0], 'k-', lw=2)
    plt.plot(k_dist, energies_TB[:, 1], 'k-', lw=2, label='TB_model')
    # dft and wannier
    dft_data = np.loadtxt('bands.dat.gnu')
    dft_kpath = dft_data[:, 0]
    dft_kpath = dft_kpath/np.max(dft_kpath)*k_dist[-1]  # scale to match wannier k-path
    dft_energies = dft_data[:, 1:]
    plt.scatter(dft_kpath, dft_energies[:, :], c='r', s=4, label='dft',zorder=2)

    wannier_data = np.loadtxt('pwscf_band.dat')
    wannier_kpath = wannier_data[:, 0]
    wannier_kpath = wannier_kpath/np.max(wannier_kpath)*k_dist[-1]  # scale to match wannier k-path
    wannier_energies = wannier_data[:, 1:]
    plt.scatter(wannier_kpath, wannier_energies[:, :], c='g', s=2, label='wannier',zorder=2)

    for pos in sym_pos:
        plt.axvline(pos, c='gray', ls='-', lw=0.5)
    plt.axhline(0, c='gray', ls='--', lw=0.5)
    
    plt.xticks(sym_pos, sym_labels)
    plt.ylabel('Energy (eV)')
    # plt.title('hBN Band Structure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(k_dist[0], k_dist[-1])
    plt.ylim(-10, 8)
    
    plt.savefig(f"wannier_bands.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    cal_hBN_bands()
    # print(np.linalg.eigvalsh(wannier_model(K)))
    # print(np.linalg.eigvalsh(TB_model(K)))