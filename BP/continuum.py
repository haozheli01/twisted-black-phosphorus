import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

class TwistedBPModel:
    def __init__(self, N_shell=1, N_top=1, N_bottom=1,
                 E_field=0.0, d_dist=0.54,
                 a_lat=3.296, b_lat=4.588,
                 delta_AA = -0.338, delta_AB = -2.912, delta_AC = 3.831, delta_AD = -0.076,
                 delta_ACp = 0.712, delta_ADp = -0.132,
                 eta_AA = 1.161, eta_AB = 2.050, eta_AC = 0.460, eta_AD = 0.104,
                 eta_ACp = -0.9765, eta_ADp = 2.699,
                 gamma_AA = -1.563, gamma_AB = 3.607, gamma_AC = -1.572, gamma_AD = 0.179,
                 gamma_ACp = 2.443, gamma_ADp = 0.364,
                 kai_AB = 3.688, kai_AC = 2.208, kai_ACp = 2.071,

                 gamma_C=0.39, gamma_V=0.07,
                 ):
        """
        Initialize TwistedBPModel with configurable parameters.
        """
        # Parameters
        self.a_lat, self.b_lat = a_lat, b_lat # lattice constants in Angstrom for BP
        self.delta_g_val = 2 * np.pi * np.abs(1/b_lat - 1/a_lat) # Moire G vector magnitude for real lattice
        self.d_dist = d_dist # effective parameter for electric field influence in bandgap in Angstrom
        self.E_field = E_field # Electric field in eV/A, seemly opposite with physical direction
        self.N_shell = N_shell # Number of G shells to include
        self.N_top = N_top # Number of top layers
        self.N_bottom = N_bottom # Number of bottom layers
        # Tight-binding parameters (from fitting to DFT)
        self.u0, self.delta, self.kai = delta_AA + delta_AD, delta_AB + delta_AC, kai_AB + kai_AC
        self.gamma_x, self.gamma_y =  eta_AB + eta_AC, gamma_AB + gamma_AC
        self.etax, self.etay = eta_AA + eta_AD, gamma_AA + gamma_AD
        self.delta_ACp, self.delta_ADp = delta_ACp, delta_ADp
        self.eta_ACp, self.eta_ADp = eta_ACp, eta_ADp
        self.gamma_ACp, self.gamma_ADp = gamma_ACp, gamma_ADp
        self.kai_ACp = kai_ACp
        self.gamma_C, self.gamma_V =  gamma_C, gamma_V # interlayer coupling parameters in eV

        # Initialize Grid
        self._init_grid()
        
        print(f"Model initialized: N_shell={N_shell}, dim_H={self.dim_H}, E_field={E_field}")

    def _init_grid(self):
        # Generate G vectors grid
        indices = np.arange(-self.N_shell, self.N_shell + 1)
        i, j = np.meshgrid(indices, indices, indexing='ij')
        
        # G vectors (Ng, 2)  !circle grid filtered by radius!
        # self.G_vectors = np.stack([i.flatten(), j.flatten()], axis=1) * self.delta_g_val

        # # G vectors (Ng, 2) !square grid ! 
        G_all = np.stack([i.flatten(), j.flatten()], axis=1) * self.delta_g_val
        G_norm = np.linalg.norm(G_all, axis=1)
        G_cut = self.N_shell * self.delta_g_val
        mask = G_norm <= G_cut + 1e-10
        self.G_vectors = G_all[mask]
        # print(self.G_vectors)

        self.dim_G = len(self.G_vectors)
        self.dim_H_top = 2 * self.dim_G * self.N_top # dimension of top layer Hamiltonian
        self.dim_H_bottom = 2 * self.dim_G * self.N_bottom # dimension of bottom layer Hamiltonian
        self.dim_H = self.dim_H_top + self.dim_H_bottom # total Hamiltonian dimension
        
        # Precompute k-independent part of Hamiltonian (Interlayer coupling)
        self.H_const = self._build_constant_hamiltonian()
        
        # Precompute k-independent generalized derivative matrices (d^2H/dk_mu dk_nu)
        self._precompute_w_matrices()

    def _build_constant_hamiltonian(self):
        """Pre-compute the part of Hamiltonian that doesn't depend on k (Hopping between layers)"""
        H = np.zeros((self.dim_H, self.dim_H), dtype=np.complex128)
        dim_block = 2 * (self.dim_H_top + self.dim_H_bottom) # total dimension of the Hamiltonian for one G
        
        # --- Diagonal in G (Interlayer Conduction) ---
        # Tc(0) <-> Bc(2) with coupling gamma_C
        idx = np.arange(self.dim_G) # num of blocks
        H[idx * dim_block + 2 * (self.dim_H_top-1) + 0, idx * dim_block + 2 * (self.dim_H_top-1) + 2] = self.gamma_C
        H[idx * dim_block + 2 * (self.dim_H_top-1) + 2, idx * dim_block + 2 * (self.dim_H_top-1) + 0] = self.gamma_C
        
        # --- Off-diagonal in G (Interlayer Valence) ---
        # Tv(1) <-> Bv(3) with coupling gamma_V
        # Connects G and G' if |G - G'| ~ delta_g and is axial
        
        # Vectorized check for neighbors
        G_diff = self.G_vectors[:, None, :] - self.G_vectors[None, :, :] # (Ng, Ng, 2)
        dists = np.linalg.norm(G_diff, axis=2)
        
        # Mask for allowed hopping
        is_neighbor = np.abs(dists - self.delta_g_val) < 1e-4
        is_axial = (np.abs(G_diff[:,:,0]) < 1e-4) | (np.abs(G_diff[:,:,1]) < 1e-4)
        mask = is_neighbor & is_axial
        
        rows, cols = np.where(mask)

        H[rows * dim_block + 2 * (self.dim_H_top-1) + 1, cols * dim_block + 2 * (self.dim_H_top-1) + 3] = self.gamma_V
        H[cols * dim_block + 2 * (self.dim_H_top-1) + 3, rows * dim_block + 2 * (self.dim_H_top-1) + 1] = self.gamma_V
        
        return H
    
    def get_hamiltonians(self, k_points):
        """
        Vectorized Hamiltonian constructor.
        
        Parameters:
        -----------
        k_points : np.ndarray
            Shape (Nk, 2)
            
        Returns:
        --------
        H_stack : np.ndarray
            Shape (Nk, dim_H, dim_H)
        """
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        u_pot = self.E_field * self.d_dist / 2.0 * (-1) # -1 for correct potential direction
        
        # Expand k to match G dimensions: k_cur is (Nk, Ng, 2)
        k_cur = k_points[:, None, :] + self.G_vectors[None, :, :]
        kx = k_cur[:, :, 0]
        ky = k_cur[:, :, 1]
        
        # Prepare base matrix from constant part
        H_stack = np.empty((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        H_stack[:] = self.H_const
        
        # Kinetic Terms (p^2)
        p2_x = kx**2
        p2_y = ky**2
        
        # Assign to diagonal blocks for all G at once
        # Indices array (Ng,) to broadcast with (Nk, Ng)
        r = np.arange(self.dim_G)

        # --- Layer (Top) ---
        for idx in range(self.N_top):
            sf = np.cos(np.pi * idx / (self.N_top + 1)) # scaling factor
            # Parameters
            val_Tc = (self.u0 + sf * self.delta_ADp) + u_pot + self.etax * p2_x + self.etay * p2_y
            val_Tv = self.u0 + u_pot + self.etax * p2_x + self.etay * p2_y
            val_T_mix = self.delta + self.gamma_x * p2_x + self.gamma_y * p2_y + 1j * self.kai * ky
        
        # --- Layer (Bottom) ---
        # Rotated 90 deg: p_x -> -p_y, p_y -> p_x
        val_Bc = self.u0 - u_pot + self.etax * p2_y + self.etay * p2_x
        val_Bv = self.u0 - u_pot + self.etax * p2_y + self.etay * p2_x
        val_B_mix = self.delta + self.gamma_x * p2_y + self.gamma_y * p2_x + 1j * self.kai * kx
        
        
        # Diagonals
        H_stack[:, r*4+0, r*4+0] += val_Tc
        H_stack[:, r*4+1, r*4+1] += val_Tv
        H_stack[:, r*4+2, r*4+2] += val_Bc
        H_stack[:, r*4+3, r*4+3] += val_Bv
        
        # Intra-layer mixing (Real)
        H_stack[:, r*4+0, r*4+1] += val_T_mix
        H_stack[:, r*4+1, r*4+0] += val_T_mix.conj()
        
        H_stack[:, r*4+2, r*4+3] += val_B_mix
        H_stack[:, r*4+3, r*4+2] += val_B_mix.conj()
        
        return H_stack

    def get_velocity_matrices(self, k_points):
        """
        Calculate velocity matrices vx, vy at k_points.
        v = dH/dk (units: eV * A)
        """
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        
        # Expand k to match G dimensions
        k_cur = k_points[:, None, :] + self.G_vectors[None, :, :]
        kx = k_cur[:, :, 0] # (Nk, Ng)
        ky = k_cur[:, :, 1]
        
        # Initialize zero matrices
        # (Nk, dim_H, dim_H)
        vx_stack = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        vy_stack = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        
        r = np.arange(self.dim_G)
        
        # --- Top Layer (alpha_x, alpha_y along x, y) ---
        # dH/dkx
        # H_00 (Tc): 
        vx_stack[:, r*4+0, r*4+0] = 2 * self.etax * kx
        # H_11 (Tv): 
        vx_stack[:, r*4+1, r*4+1] = 2 * self.etax * kx
        # H_01/10 (Mix):
        vx_stack[:, r*4+0, r*4+1] = 2 * self.gamma_x * kx + 1j * self.kai * 0
        vx_stack[:, r*4+1, r*4+0] = 2 * self.gamma_x * kx - 1j * self.kai * 0
        
        # dH/dky
        # H_00 (Tc): 
        vy_stack[:, r*4+0, r*4+0] = 2 * self.etay * ky
        # H_11 (Tv): 
        vy_stack[:, r*4+1, r*4+1] = 2 * self.etay * ky
        # Mixing term in y
        vy_stack[:, r*4+0, r*4+1] = 2 * self.gamma_y * ky + 1j * self.kai
        vy_stack[:, r*4+1, r*4+0] = 2 * self.gamma_y * ky - 1j * self.kai

        # --- Bottom Layer (Rotated: params swapped effectively) ---
        # dH/dkx
        # H_22 (Bc):
        vx_stack[:, r*4+2, r*4+2] = 2 * self.etay * kx
        # H_33 (Bv):
        vx_stack[:, r*4+3, r*4+3] = 2 * self.etay * kx
        # Mixing
        vx_stack[:, r*4+2, r*4+3] = 2 * self.gamma_y * kx + 1j * self.kai
        vx_stack[:, r*4+3, r*4+2] = 2 * self.gamma_y * kx - 1j * self.kai
        
        # dH/dky
        # H_22 (Bc):
        vy_stack[:, r*4+2, r*4+2] = 2 * self.etax * ky
        # H_33 (Bv): 
        vy_stack[:, r*4+3, r*4+3] = 2 * self.etax * ky
        # Mixing: 
        vy_stack[:, r*4+2, r*4+3] = 2 * self.gamma_x * ky + 1j * self.kai * 0
        vy_stack[:, r*4+3, r*4+2] = 2 * self.gamma_x * ky - 1j * self.kai * 0
        
        return vx_stack, vy_stack

    def _precompute_w_matrices(self):
        """Pre-compute k-independent generalized derivative matrices (d^2H/dk_mu dk_nu)."""
        r = np.arange(self.dim_G)
        
        w_xx = np.zeros((self.dim_H, self.dim_H), dtype=np.complex128)
        w_yy = np.zeros((self.dim_H, self.dim_H), dtype=np.complex128)
        w_xy = np.zeros((self.dim_H, self.dim_H), dtype=np.complex128)
        
        # --- Top Layer ---
        w_xx[r*4+0, r*4+0] = 2 * self.etax
        w_yy[r*4+0, r*4+0] = 2 * self.etay
        w_xx[r*4+1, r*4+1] = 2 * self.etax
        w_yy[r*4+1, r*4+1] = 2 * self.etay
        
        # --- Bottom Layer (Rotated 90 deg: alpha_x <-> alpha_y) ---
        w_xx[r*4+2, r*4+2] = 2 * self.etay
        w_yy[r*4+2, r*4+2] = 2 * self.etax
        w_xx[r*4+3, r*4+3] = 2 * self.etay
        w_yy[r*4+3, r*4+3] = 2 * self.etax
        
        # --- Mixing Terms ---
        w_xx[r*4+0, r*4+1] = 2 * self.gamma_x
        w_xx[r*4+1, r*4+0] = 2 * self.gamma_x
        w_yy[r*4+0, r*4+1] = 2 * self.gamma_y
        w_yy[r*4+1, r*4+0] = 2 * self.gamma_y
        w_xx[r*4+2, r*4+3] = 2 * self.gamma_y
        w_xx[r*4+3, r*4+2] = 2 * self.gamma_y
        w_yy[r*4+2, r*4+3] = 2 * self.gamma_x
        w_yy[r*4+3, r*4+2] = 2 * self.gamma_x
        
        self.w_xx_const = w_xx
        self.w_yy_const = w_yy
        self.w_xy_const = w_xy

    def get_generalized_derivative_matrices(self, k_points):
        """
        w_munu = d^2H / dk_mu dk_nu
        Returns operator matrices w_xx, w_yy, w_xy representing the curvature of the Hamiltonian.
        Tiles the precomputed constant matrices to match the number of k-points.
        """
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        
        w_xx = np.broadcast_to(self.w_xx_const, (Nk, self.dim_H, self.dim_H))
        w_yy = np.broadcast_to(self.w_yy_const, (Nk, self.dim_H, self.dim_H))
        w_xy = np.broadcast_to(self.w_xy_const, (Nk, self.dim_H, self.dim_H))

        return w_xx, w_yy, w_xy

def cal_bands(N_shell=1, E_field=0.0, k_fine_steps=360, 
                   y_lim=(-1.0, 1.0), save_prefix=""):
    
    # 1. Initialize Model
    model = TwistedBPModel(N_shell=N_shell, E_field=E_field)
    
    # 2. Physics Constants & Supercell
    dg = model.delta_g_val
    G_super = dg / 2.0
    k_boundary = G_super / 2.0
    
    # 3. Path Generation
    n_seg = k_fine_steps // 2
    
    # Path 1: X -> Gamma (kx: -dg/2 -> 0)
    path_1 = np.zeros((n_seg, 2))
    path_1[:, 0] = np.linspace(-dg/2, 0, n_seg, endpoint=False)
    
    # Path 2: Gamma -> Y (ky: 0 -> dg/2)
    path_2 = np.zeros((n_seg + 1, 2))
    path_2[:, 1] = np.linspace(0, dg/2, n_seg + 1)
    
    k_path = np.vstack([path_1, path_2])
    
    # Calculate path distances for plotting
    dists = np.linalg.norm(np.diff(k_path, axis=0), axis=1)
    k_dist = np.concatenate([[0], np.cumsum(dists)])
    
    # High symmetry markers
    k_high_sym_pos = [0.0, k_dist[n_seg], k_dist[-1]]
    k_high_sym_labels = [r'$X$', r'$\Gamma$', r'$Y$']
    
    # 4. Unfolded Bands Calculation (Vectorized)
    print(f"Calculating unfolded bands ({len(k_path)} k-points)...")
    H_list = model.get_hamiltonians(k_path)
    # eigvalsh broadcasts over the first dimension (stacks of matrices)
    unfolded_energies = np.linalg.eigvalsh(H_list)
    
    # 5. Folded Bands Calculation (Vectorized Scanning)
    print(f"Calculating folded bands...")
    folded_k_dist = []
    folded_energies = []
    
    n_scan = 2 * n_seg + 1
    tol = 1e-5
    
    # --- X Scan ---
    x_scan_vals = np.linspace(-dg/2, dg/2, n_scan)
    k_scan_x = np.zeros((n_scan, 2))
    k_scan_x[:, 0] = x_scan_vals
    
    H_x = model.get_hamiltonians(k_scan_x)
    E_x = np.linalg.eigvalsh(H_x)
    
    # Fold Logic X
    kx_folded = ((x_scan_vals + k_boundary) % G_super) - k_boundary
    # We want points that fold into [-k_boundary, 0] (X' -> Gamma)
    mask_x = (kx_folded >= -k_boundary - tol) & (kx_folded <= tol)
    
    # Map to plot: X' map to 0, Gamma map to k_boundary
    # Distance = k_boundary + kx_folded
    pos_x = k_boundary + kx_folded[mask_x]
    valid_E_x = E_x[mask_x]
    
    # Flatten and append
    if len(pos_x) > 0:
        # Repeat pos for each band
        N_bands = valid_E_x.shape[1]
        folded_k_dist.extend(np.repeat(pos_x, N_bands))
        folded_energies.extend(valid_E_x.flatten())

    # --- Y Scan ---
    y_scan_vals = np.linspace(-dg/2, dg/2, n_scan)
    k_scan_y = np.zeros((n_scan, 2))
    k_scan_y[:, 1] = y_scan_vals
    
    H_y = model.get_hamiltonians(k_scan_y)
    E_y = np.linalg.eigvalsh(H_y)
    
    # Fold Logic Y
    ky_folded = ((y_scan_vals + k_boundary) % G_super) - k_boundary
    # We want points that fold into [0, k_boundary] (Gamma -> Y')
    mask_y = (ky_folded >= -tol) & (ky_folded <= k_boundary + tol)
    
    # Map to plot: Gamma map to k_boundary, Y' map to 2*k_boundary
    # Distance = k_boundary + ky_folded
    pos_y = k_boundary + ky_folded[mask_y]
    valid_E_y = E_y[mask_y]
    
    if len(pos_y) > 0:
        N_bands = valid_E_y.shape[1]
        folded_k_dist.extend(np.repeat(pos_y, N_bands))
        folded_energies.extend(valid_E_y.flatten())
    
    folded_k_dist = np.array(folded_k_dist)
    folded_energies = np.array(folded_energies)

    # 6. Plotting
    plot_2D_bands(k_dist, unfolded_energies, folded_k_dist, folded_energies,
                  k_high_sym_pos, k_high_sym_labels,
                  k_boundary, y_lim, suffix=f"E{E_field:.3f}{save_prefix}")
                  
    return model

def plot_2D_bands(k_dist, unfolded_E, folded_k, folded_E, 
                  sym_pos, sym_labels, k_boundary, y_lim, suffix):
    
    # Unfolded Plot
    plt.figure(figsize=(8, 8))

    # # load vasp plot data
    # vasp_dat = np.loadtxt("341.dat")
    # vasp_kpath = vasp_dat[:, 0]
    # vasp_kpath = vasp_kpath / max(vasp_kpath) * k_dist[-1] # normalize to our k_dist
    # vasp_energies = vasp_dat[:, 1] + 0.223446 # set VBM to zero

    # Vectorized plotting of lines is faster than loop but mpl handles loop ok.
    # Plotting first band with label
    nbnd = unfolded_E.shape[-1]
    vbm = np.max(unfolded_E[:, :nbnd//2]) # Assuming half filling, VBM is max of valence bands
    
    plt.plot(k_dist, unfolded_E[:, 0] - vbm, 'b-', lw=2.5, alpha=0.5, label='Unfolded Bands')
    plt.plot(k_dist, unfolded_E[:, 1:] - vbm, 'b-', lw=2.5, alpha=0.5)
    # plt.scatter(vasp_kpath, vasp_energies, s=20, color='black', alpha=0.8, label='VASP Bands',
    #             facecolors='white', edgecolors='black', linewidths=1.6)
    for pos in sym_pos:
        plt.axvline(pos, c='gray', ls='-', lw=0.5)
    plt.xticks(sym_pos, sym_labels)
    plt.ylim(y_lim)
    plt.xlim(k_dist[0], k_dist[-1])
    plt.ylabel("Energy (eV)")
    # plt.title(f"Unfolded Band Structure{suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"kp_{suffix}.png", dpi=200)
    plt.close()

    # Folded Plot
    plt.figure(figsize=(8, 8))
    
    # load vasp plot data
    vasp_dat = np.loadtxt("571.dat")
    vasp_kpath = vasp_dat[:, 0]
    vasp_kpath = vasp_kpath / max(vasp_kpath) * max(folded_k) # normalize to our k_dist
    vasp_energies = vasp_dat[:, 1] + 0.271584 # set VBM to zero

    plt.scatter(vasp_kpath, vasp_energies, s=20, color='black', alpha=0.8, label='VASP Bands',
                facecolors='white', edgecolors='black', linewidths=1.6)

    folded_sym_pos = [0.0, k_boundary, 2*k_boundary]
    folded_sym_labels = [r"$X'$", r'$\Gamma$', r"$Y'$"]
    
    plt.scatter(folded_k, folded_E - vbm, s=20, color='red', alpha=0.9, label='Folded Bands',
                facecolors='white', edgecolors='red', linewidths=0.5, zorder=0)
    
    for pos in folded_sym_pos:
        plt.axvline(pos, c='gray', ls='-', lw=0.5)
    plt.xticks(folded_sym_pos, folded_sym_labels)
    plt.ylim(y_lim)
    plt.xlim(0, 2*k_boundary)
    plt.ylabel("Energy (eV)")
    # plt.title(f"Folded Band Structure{suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"Folded_kp_{suffix}.png", dpi=200)
    plt.close()
    
    print(f"Figures saved with suffix {suffix}")

def plot_3d_bands(N_shell=1, E_field=0.0, k_range=0.2, n_grid=40, 
                  bands_to_plot=4, view_elev=30, view_azim=45, save_prefix=""):
    """
    Plot 3D band structure surface for the unfolded Hamiltonian on a 2D k-grid.
    
    Parameters:
    -----------
    N_shell : int
        Number of shells
    E_field : float
        Electric field
    k_range : float
        K-space range [-k_range, k_range] for both kx and ky (centered at Gamma)
    n_grid : int
        Grid density (n_grid x n_grid)
    bands_to_plot : int
        Number of bands to plot around the gap (must be even). 
        e.g. 4 means 2 valence and 2 conduction bands.
    """
    print(f"Generating 3D band plot with range [{-k_range}, {k_range}]...")
    
    model = TwistedBPModel(N_shell=N_shell, E_field=E_field)
    
    # Generate 2D grid
    kx = np.linspace(-k_range, k_range, n_grid)
    ky = np.linspace(-k_range, k_range, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    
    # Flatten for vectorized calculation
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    
    # Calculate Hamiltonians
    H_list = model.get_hamiltonians(k_points)
    
    # Eigenvalues
    evals = np.linalg.eigvalsh(H_list) # Shape (n_grid^2, n_bands)
    
    # Reshape back to grid
    n_bands = evals.shape[1]
    E_grid = evals.reshape(n_grid, n_grid, n_bands)
    
    # Find gap / separation index
    # Assuming gap is around the middle of bands for this model or around 0 eV
    # Let's find index where energy crosses 0 or just take middle bands
    # The model has 4*Dim_G bands. With half filling usually.
    # We'll just plot the middle bands.
    mid_idx = n_bands // 2
    start_band = max(0, mid_idx - bands_to_plot // 2)
    end_band = min(n_bands, mid_idx + bands_to_plot // 2)
    
    fig = plt.figure(figsize=(8.0, 8.0))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surfaces
    for b in range(start_band, end_band):
        Z = E_grid[:, :, b]
        # Color based on band index to distinguish
        if b < mid_idx:
            cmap = 'viridis' # Valence
            alpha = 0.9
        else: 
            cmap = 'plasma' # Conduction
            alpha = 0.9
            
        surf = ax.plot_surface(KX, KY, Z, cmap=cmap, alpha=alpha, 
                               edgecolor='none', antialiased=False)
        
    ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    ax.set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    ax.set_zlabel('Energy (eV)')
    ax.set_title(f'3D Band Structure')
    
    # Adjust aspect ratio (x, y, z) - Stretch Z axis relative to XY
    ax.set_box_aspect((1, 1, 1.5))
    
    # Set init view
    ax.view_init(elev=view_elev, azim=view_azim)
    
    suffix = f"_3D_E{E_field:.3f}{save_prefix}.png"
    plt.tight_layout()
    plt.savefig(f"kp{suffix}", dpi=300)
    plt.close()
    
    print(f"Saved 3D plot: kp{suffix}")

def calculate_optical_conductivity(N_shell=1, E_field=0.0, 
                                   E_range=(0.0, 1.0), n_E=500, eta=0.010,
                                   k_range=0.15, n_k=60, save_prefix=""):
    """
    Calculate the interband optical response using velocity matrix elements,
    and construct an absorption-like spectrum.

    This routine implements the interband part of the Kubo-Greenwood formula
    in the velocity gauge. The central quantity computed is a spectral function
    proportional to the imaginary part of the dielectric function ε₂(ω),
    from which an absorption-like spectrum is constructed.

    Specifically, we evaluate (up to overall constants):

        \epsilon_2^{ii}(\omega) \propto
        \sum_{v,c} \int_{BZ} d^2k
        |\langle c,k | v_i | v,k \rangle|^2 / (E_{c,k} - E_{v,k})^2
        \cdot \delta(E_{c,k} - E_{v,k} - \hbar \omega)

    where:
        - v_i is the velocity operator (i = x, y),
        - v, c label valence and conduction bands,
        - the factor (E_c - E_v)^{-2} arises from converting the
          velocity-gauge transition rate into an electric-field response
          (i.e. from A to E, equivalent to length gauge).

    The delta function is approximated by a Lorentzian broadening with width \eta.

    The final plotted quantity is proportional to \omega \cdot \epsilon_2(\omega), which is commonly
    used as an absorption spectrum (up to material-dependent prefactors).
    Absolute units are not enforced.

    Parameters
    ----------
    N_shell : int
        Number of shells in the tight-binding model.
    E_field : float
        External electric field strength.
    E_range : tuple of float
        Photon energy range (eV).
    n_E : int
        Number of photon energy points.
    eta : float
        Lorentzian broadening parameter (eV).
    k_range : float
        Integration range in k-space: [-k_range, k_range].
    n_k : int
        Number of k points per direction (n_k x n_k grid).
    save_prefix : str
        Optional suffix for saved figure filenames.
    """

    print(f"Calculating optical conductivity spectrum...")
    model = TwistedBPModel(N_shell=N_shell, E_field=E_field)
    
    # 1. Generate K-grid
    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)
    
    # 2. Diagonalize Hamiltonian
    print(f"  Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack) # eigh for Hermitian
    # evals: (Nk, Nb), evecs: (Nk, Nb, Nb) (columns are eigenvectors)
    
    # 3. Calculate Velocity Matrices
    print(f"  Calculating velocity matrices...")
    vx_stack, vy_stack = model.get_velocity_matrices(k_points)
    
    # 4. Compute Matrix Elements <n|v|m>
    # Transform v to eigenbasis: v_nm = U^dag * v * U
    # evecs[k] is U
    # We want: v_mat[k] = evecs[k].conj().T @ v_stack[k] @ evecs[k]
    # Vectorized matmul:
    
    # Conjugate transpose of eigenvectors
    U_dag = np.conj(np.transpose(evecs, (0, 2, 1)))
    
    # v_x matrix in eigenbasis (Nk, Nb, Nb)
    vx_eig = U_dag @ vx_stack @ evecs
    vy_eig = U_dag @ vy_stack @ evecs
    
    # We only need magnitudes squared
    Mx2 = np.abs(vx_eig)**2
    My2 = np.abs(vy_eig)**2
    
    # 5. Compute Sigma(omega)
    omegas = np.linspace(E_range[0], E_range[1], n_E)
    sigma_xx = np.zeros_like(omegas)
    sigma_yy = np.zeros_like(omegas)
    
    n_bands = evals.shape[1]
    mid_idx = n_bands // 2 
    # Assume half-filling, so transitions from valence (bands < mid_idx) to conduction (bands >= mid_idx)
    # Actually, we should check occupation properly, but for this model it's usually insulating or semimetallic at charge neutrality.
    # We sum over all pairs i -> j with E_j > E_i
    
    print(f"  Summing transitions...")
    
    # Vectorized over all valence-conduction band pairs
    # Extract valence and conduction blocks of matrix elements
    # Mx2, My2: (Nk, Nb, Nb). We need Mx2[:, i, j] for i in [0,mid), j in [mid,Nb)
    Mx2_vc = Mx2[:, :mid_idx, mid_idx:]  # (Nk, Nv, Nc)
    My2_vc = My2[:, :mid_idx, mid_idx:]  # (Nk, Nv, Nc)
    
    # Energy differences for all v-c pairs: (Nk, Nv, Nc)
    delta_E_all = evals[:, mid_idx:, None] - evals[:, None, :mid_idx]  # (Nk, Nc, Nv)
    delta_E_all = delta_E_all.transpose(0, 2, 1)  # (Nk, Nv, Nc)
    
    # |<c|v_i|v>|^2 / (Delta E)^2
    #
    # NOTE:
    # The (ΔE)^{-2} factor does NOT belong to the bare transition rate.
    # It appears here because we are constructing \epsilon_2(\omega) (or an absorption-like
    # response), i.e. an electric-field response function.
    #
    # Using \hbar \omega = \Delta E, this factor accounts for the conversion from vector
    # potential A to electric field E, and makes the result equivalent
    # to the length-gauge formulation.
    M_weighted_x = Mx2_vc / delta_E_all**2  # (Nk, Nv, Nc)
    M_weighted_y = My2_vc / delta_E_all**2
    
    # Flatten pairs dimension: (Nk, Nv*Nc)
    N_pairs = mid_idx * (n_bands - mid_idx)
    M_flat_x = M_weighted_x.reshape(Nk, N_pairs)
    M_flat_y = M_weighted_y.reshape(Nk, N_pairs)
    dE_flat = delta_E_all.reshape(Nk, N_pairs)
    
    # Process in batches to limit memory: (n_E, Nk, batch) Lorentzian
    batch_size = max(1, min(N_pairs, max(1, 200_000_000 // (n_E * Nk))))
    for b_start in range(0, N_pairs, batch_size):
        b_end = min(b_start + batch_size, N_pairs)
        dE_batch = dE_flat[:, b_start:b_end]       # (Nk, batch)
        Mx_batch = M_flat_x[:, b_start:b_end]      # (Nk, batch)
        My_batch = M_flat_y[:, b_start:b_end]
        
        # Lorentzian: (n_E, Nk, batch)
        diff = omegas[:, None, None] - dE_batch[None, :, :]
        lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
        
        # Sum over k and batch dims
        sigma_xx += np.sum(lorentz * Mx_batch[None, :, :], axis=(1, 2))
        sigma_yy += np.sum(lorentz * My_batch[None, :, :], axis=(1, 2))

    # Normalize by number of k-points (density)
    sigma_xx /= Nk
    sigma_yy /= Nk

    # --- Construct an absorption-like spectrum ---
    #
    # The quantity accumulated above is proportional to \epsilon_2(\omega).
    # A commonly plotted absorption spectrum is proportional to \omega \cdot \epsilon_2(\omega),
    # up to material- and unit-dependent prefactors.
    #
    # Here we multiply by \omega to obtain a quantity suitable for qualitative
    # comparison of optical absorption features.
    absorption_xx = np.zeros_like(sigma_xx)
    absorption_yy = np.zeros_like(sigma_yy)
    
    absorption_xx = sigma_xx * omegas
    absorption_yy = sigma_yy * omegas
    
    # 6. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(omegas, absorption_xx, 'r-', label=r'x-polarized', lw=2)
    plt.plot(omegas, absorption_yy, 'b--', label=r'y-polarized', lw=2)
    
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Optical Absorption (a.u.)')
    plt.title(f'E={E_field} eV/A, $\eta$={eta*1000:.1f} meV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)
    # plt.ylim(bottom=0)
    
    suffix = f"_Abs_E{E_field:.3f}{save_prefix}.png"
    plt.savefig(f"kp{suffix}", dpi=300)
    plt.close()
    
    print(f"Saved Optical Absorption Spectrum: kp{suffix}")

def plot_transition_matrix_elements(N_shell=1, E_field=0.0, 
                                    band_indices=None,
                                    k_range=0.15, n_k=60, save_prefix=""):
    """
    Plot the magnitude squared of transition matrix elements |<j|v|i>|^2 
    in k-space as contour plots.
    
    Parameters:
    -----------
    band_indices : tuple or None
        (initial_band, final_band) indices (0-based).
        If None, defaults to transition between Valence Top and Conduction Bottom.
    """
    print(f"Calculating transition matrix elements map...")
    model = TwistedBPModel(N_shell=N_shell, E_field=E_field)
    
    # Defaults
    if band_indices is None:
        # Assuming half-filling
        mid = model.dim_H // 2
        band_i = mid - 1 # VBM
        band_j = mid     # CBM
    else:
        band_i, band_j = band_indices
        
    print(f"  Mapping transition: Band {band_i} -> Band {band_j}")   # 0-based
    
    # 1. Generate Grid
    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    
    # 2. Diagonalize
    H_stack = model.get_hamiltonians(k_points)
    _, evecs = np.linalg.eigh(H_stack)
    
    # 3. Velocity
    vx_stack, vy_stack = model.get_velocity_matrices(k_points)
    
    # 4. Matrix Elements (Specific bands only to save memory/time)
    # We want <j|v|i> = u_j^dag @ v @ u_i
    # evecs shape: (Nk, dim_H, dim_H). Columns are eigenvectors.
    u_i = evecs[:, :, band_i] # (Nk, dim_H)
    u_j = evecs[:, :, band_j] # (Nk, dim_H)
    
    
    # X - component
    # v @ u_i : (Nk, H, H) @ (Nk, H, 1) -> (Nk, H) if we reshape u_i
    # Let's use einsum for clarity
    # k: k-point index, a: row index, b: col index
    
    M_x = np.einsum('ka,kab,kb->k', u_j.conj(), vx_stack, u_i)
    M_y = np.einsum('ka,kab,kb->k', u_j.conj(), vy_stack, u_i)
    
    M_x2 = np.abs(M_x)**2
    M_y2 = np.abs(M_y)**2
    
    # Reshape to grid
    Z_x = M_x2.reshape(n_k, n_k)
    Z_y = M_y2.reshape(n_k, n_k)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Shared limits and levels
    zs = np.concatenate([Z_x.flatten(), Z_y.flatten()])
    vmax = np.percentile(zs, 99) # Cut off outliers
    # vmax = max(zs.max(), 1e-6)
    
    # Plot X
    c1 = axes[0].contourf(KX, KY, Z_x, levels=40, cmap='plasma', vmin=0, vmax=vmax)
    axes[0].set_title(r'$\mathregular{|\langle \psi_f | v_x | \psi_i \rangle|^2}$ (x-pol)')
    axes[0].set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    axes[0].set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    axes[0].set_aspect('equal')
    plt.colorbar(c1, ax=axes[0])
    
    # Plot Y
    c2 = axes[1].contourf(KX, KY, Z_y, levels=40, cmap='plasma', vmin=0, vmax=vmax)
    axes[1].set_title(r'$\mathregular{|\langle \psi_f | v_y | \psi_i \rangle|^2}$ (y-pol)')
    axes[1].set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    axes[1].set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    axes[1].set_aspect('equal')
    plt.colorbar(c2, ax=axes[1])
    
    plt.suptitle(f"Transition Matrix Elements: Band {band_i} -> {band_j}\n(E={E_field} eV/A)") # 0-based
    plt.tight_layout()
    
    suffix = f"_M_E{E_field:.3f}_B{band_i}-{band_j}{save_prefix}.png" # 0-based
    plt.savefig(f"kp{suffix}", dpi=300)
    plt.close()
    print(f"Saved Matrix Element Map: kp{suffix}")

def calculate_shift_current(N_shell=1, E_field=0.0,
                            E_range=(0.0, 1.0), n_E=400, eta=0.010,
                            k_range=0.15, n_k=60, band_window=None):
    r"""
    Shift current sigma^{abc}(ω) via gauge-invariant Sum-Over-States method.

    Ref: Phys. Rev. B 61, 5337 (2000)

        \\sigma^{abc}(\omega) = C * \\sum_{nm}[ f_{nm} \\Im[r^b_{mn} (r^c_{nm})_{;a}] * \\delta(\omega - \omega_{mn})]

    where:
        r^b_{mn} = v^b_{mn} / (i \\omega_{mn})
        (r^c_{nm})_{;a} = (-1/i\\omega_{nm}) [ term_A/\\omega_{nm} + term_B + term_C ]
        term_A = v^c_{nm} \\delta^a_{nm} + v^a_{nm} \\delta^c_{nm}       (\\delta^a_{nm} = v^a_{nn} - v^a_{mm})
        term_B = \\sum_{p\\neq n,m} [v^c_{np} v^a_{pm}/\\omega_{pm} - v^a_{np} v^c_{pm}/\\omega_{np}]
        term_C = - v^{ac}_{nm}  (generalized derivative of velocity)

    In the moire BP model, term_B is generally nonzero due to multiband couplings.

    Parameters:
        comp : (a, b, c) — tensor component directions
        band_window : (v_start, v_end, c_start, c_end) or None
        model : optional pre-built TwistedBPModel
    """
    comp_list=[('x', 'x', 'x'), ('x', 'y', 'y'), ('y', 'x', 'x'), ('y', 'y', 'y')]
    plt.figure(figsize=(8, 6))

    # --- Shared computation for all components (done once) ---
    model = TwistedBPModel(N_shell=N_shell, E_field=E_field)
    
    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack) 
    Nb = evals.shape[1]

    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    w_xx, w_yy, w_xy = model.get_generalized_derivative_matrices(k_points)
    
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))

    def to_eig(O):
        return U_dag @ O @ U

    v_map = {'x': to_eig(vx_orb), 'y': to_eig(vy_orb)}
    w_map = {'xx': to_eig(w_xx), 'yy': to_eig(w_yy), 'xy': to_eig(w_xy), 'yx': to_eig(w_xy)}

    # Band selection
    mid = Nb // 2
    if band_window is None:
        v_idx = np.arange(0, mid)
        c_idx = np.arange(mid, Nb)
    else:
        v_idx = np.arange(band_window[0], band_window[1]+1)
        c_idx = np.arange(band_window[2], band_window[3]+1)

    # Precompute all energy differences for Term B (shared across components)
    eps_denom = 1e-5
    # w_all[k, a, b] = E_a - E_b
    w_all = evals[:, :, None] - evals[:, None, :]  # (Nk, Nb, Nb)

    omegas = np.linspace(E_range[0], E_range[1], n_E)
    results = {}

    for comp in comp_list:
        a_dir, b_dir, c_dir = comp
        print(f"Calculating shift current sigma^{{{a_dir}{b_dir}{c_dir}}}(omega) "
            f"(n_k={n_k}, η={eta*1000:.0f} meV)...")

        v_a = v_map[a_dir]
        v_b = v_map[b_dir]
        v_c = v_map[c_dir]
        w_ac_key = a_dir + c_dir
        w_ac = w_map[w_ac_key]

        # Spectrum
        sigma = np.zeros_like(omegas)
        
        print(f"  Transitions: {len(v_idx)} val x {len(c_idx)} cond, Nk={Nk}")

        for n in v_idx:
            f_n = 1.0 if n < mid else 0.0
            
            # Precompute n-row velocity slices once per n
            v_c_n_row = v_c[:, n, :]  # (Nk, Nb)
            v_a_n_row = v_a[:, n, :]  # (Nk, Nb)
            v_a_nn = v_a[:, n, n]     # (Nk,)
            v_c_nn = v_c[:, n, n]     # (Nk,)
            w_n_all = w_all[:, n, :]  # (Nk, Nb)  w_np = E_n - E_p
            
            for m in c_idx:
                f_m = 1.0 if m < mid else 0.0
                f_nm = f_n - f_m
                if f_nm == 0.0:
                    continue

                w_mn = evals[:, m] - evals[:, n]  # (Nk,)
                nonzero = w_mn > eps_denom

                r_b_mn = np.zeros(Nk, dtype=np.complex128)
                r_b_mn[nonzero] = v_b[nonzero, m, n] / (1j * w_mn[nonzero])

                # Term A
                termA = np.zeros(Nk, dtype=np.complex128)
                delta_a = v_a_nn[nonzero] - v_a[nonzero, m, m]
                delta_c = v_c_nn[nonzero] - v_c[nonzero, m, m]
                termA[nonzero] = (v_c[nonzero, n, m] * delta_a
                                + v_a[nonzero, n, m] * delta_c) / (-w_mn[nonzero])

                # Term B — use precomputed energy differences
                w_np = w_n_all          # (Nk, Nb)
                w_pm = w_all[:, :, m]   # (Nk, Nb)  w_pm = E_p - E_m
                valid_p = (np.abs(w_np) > eps_denom) & (np.abs(w_pm) > eps_denom)
                valid_p[:, n] = False;  valid_p[:, m] = False
                valid_p &= nonzero[:, None]

                v_a_col_m = v_a[:, :, m]  # (Nk, Nb)
                v_c_col_m = v_c[:, :, m]  # (Nk, Nb)
                num1 = v_c_n_row * v_a_col_m
                num2 = v_a_n_row * v_c_col_m
                termB_contrib = np.zeros((Nk, Nb), dtype=np.complex128)
                termB_contrib[valid_p] = (num1[valid_p] / w_pm[valid_p]
                                        - num2[valid_p] / w_np[valid_p])
                termB = np.sum(termB_contrib, axis=1)

                # Term C
                termC = -w_ac[:, n, m]

                K_nm = termA + termB + termC
                r_deriv = np.zeros(Nk, dtype=np.complex128)
                r_deriv[nonzero] = K_nm[nonzero] / (-1j * (-w_mn[nonzero]))

                weight = f_nm * np.imag(r_b_mn * r_deriv)
                diff = omegas[:, None] - w_mn[None, :]
                lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
                sigma += np.sum(lorentz * weight[None, :], axis=1)

        # Normalize by k-point count
        sigma /= Nk

        # Physical prefactor: sigma has units of Å² so far (from velocity = dH/dk in eV·Å).
        # Full formula: sigma^{abc} = (2*pi*e^2) / (hbar * A_uc) * I(omega)
        # with an extra 1/hbar absorbed into the Lorentzian (delta in 1/eV → 1/(eV·s^{-1}))
        e_charge = 1.602176634e-19   # C
        hbar = 1.054571817e-34       # J·s
        A_uc = model.a_lat * model.b_lat  # Å² (orthorhombic unit cell)
        prefactor = (2 * np.pi * e_charge**2) / (hbar * A_uc) * 1E6  # → μA·Å / V²
        sigma *= prefactor
        results[comp] = sigma

    # --- Plotting ---
    plt.plot(omegas, results[('x', 'x', 'x')], 'r-', lw=2, label=fr'$\sigma^{{xxx}}(\omega)$')
    plt.plot(omegas, results[('y', 'y', 'y')], 'b--', lw=2, label=fr'$\sigma^{{yyy}}(\omega)$')
    plt.plot(omegas, results[('y', 'x', 'x')], 'g-', lw=2, label=fr'$\sigma^{{yxx}}(\omega)$')
    plt.plot(omegas, results[('x', 'y', 'y')], 'm--', lw=2, label=fr'$\sigma^{{xyy}}(\omega)$')

    plt.axhline(0, color='k', lw=0.5, ls='--')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel(r'Shift Conductivity ($\mu$A$\cdot$Å/V$^2$)')
    plt.title(f'Shift Current Spectrum \nE={E_field} eV/A')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)
    
    fname = f"kp_sc.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    
    print(f"Saved Shift Current Figure")
    return omegas, results

if __name__ == "__main__":
    # BP parameters
    a_lat=4.588
    b_lat=3.296
    G_moire = 2 * np.pi * np.abs(1/b_lat - 1/a_lat) # Moiré G vector magnitude for real lattice

    # Calculate Area
    area_uc = 2 * a_lat * b_lat # Approximate factor (depends on supercell definition)
    
    # single k point test
    # --------------------------------------------
    # model = TwistedBPModel(N_shell=1, E_field=0.0)
    # k_test = np.array([[0.0, 0.0]])
    # H_test = model.get_hamiltonians(k_test)
    # print(H_test[0])
    # print("Eigenvalues at Gamma:", np.linalg.eigvalsh(H_test[0]))

    # Standard 2D Band Structure
    # --------------------------------------------
    cal_bands(N_shell=1, E_field=0.0, k_fine_steps=200, y_lim=(-0.5,2))
    
    # 3D Band Structure
    # --------------------------------------------
    # plot_3d_bands(N_shell=1, E_field=0.0, k_range=G_moire/2, n_grid=60, bands_to_plot=4,
    #               view_elev=15, view_azim=45)
                  
    # Optical Conductivity
    # --------------------------------------------
    # calculate_optical_conductivity(N_shell=1, E_field=0.0, n_k=160, n_E=500,
    #                                eta=0.010, k_range=G_moire/2, E_range=(0.0, 2.0))
                                   
    # # Matrix Element Map (VBM -> CBM)
    # --------------------------------------------
    # plot_transition_matrix_elements(N_shell=1, E_field=0.0, 
    #                                 # band_indices=(1, 2),
    #                                 k_range=G_moire/2, n_k=160)
    
    # # Shift Current Calculation
    # --------------------------------------------   
    calculate_shift_current(N_shell=2, E_field=0.0, n_k=120, n_E=100,
                            # band_window=(8,9,10,10), 
                            E_range=(0.0, 3.0), k_range=G_moire/2,)