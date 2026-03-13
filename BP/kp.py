import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

class TwistedBPModel:
    def __init__(self, N_shell=1, E_field=0.0, 
                 a_lat=4.588, b_lat=3.296,

                # # Best fit parameters for 341 supercell, different from unitcell's parameters
                #  gamma_C=0.224, gamma_V=0.06,
                #  e_C0=0.224 + 0.615, e_V0=0.224 -0.853 + 0.615,
                #  gamma_ML=3.450, beta_ML=0.00, 
                #  alpha_Cx=1.100, alpha_Vx=-3.600,
                #  alpha_Cy=2.800, alpha_Vy=-1.800,
                #  d_dist=0.54

                # # Parameters for 571 supercell, from Ref: 2D Mater. 4 (2017) 035025
                 gamma_C=0.33, gamma_V=0.07,
                 e_C0=0.33 + 0.579, e_V0=0.33 -0.927 + 0.579,
                 gamma_ML=4.50, beta_ML=0.00, 
                 alpha_Cx=1.20, alpha_Vx=-5.90,
                 alpha_Cy=2.70, alpha_Vy=-2.00,
                 d_dist=0.54

                # # Best fit parameters for 571 supercell, different from unitcell's parameters
                #  gamma_C=0.30, gamma_V=0.065,
                #  e_C0=0.30 + 0.579, e_V0=0.30 -0.897 + 0.579,
                #  gamma_ML=4.50, beta_ML=0.00, 
                #  alpha_Cx=1.500, alpha_Vx=-4.900,
                #  alpha_Cy=2.800, alpha_Vy=-1.800,
                #  d_dist=0.54
                 ):
        """
        Initialize TwistedBPModel with configurable parameters.
        """
        # Parameters
        self.a_lat, self.b_lat = a_lat, b_lat # lattice constants in Angstrom for BP
        self.delta_g_val = 2 * np.pi * np.abs(1/b_lat - 1/a_lat) # Moire G vector magnitude for real lattice
        self.gamma_ML, self.beta_ML = gamma_ML, beta_ML # intralayer coupling parameters in eV of monolayer BP
        self.gamma_C, self.gamma_V =  gamma_C, gamma_V # interlayer coupling parameters in eV
        self.e_C0, self.e_V0 = e_C0, e_V0 # band edge energies in eV, using monolayer values
        self.alpha_Cx, self.alpha_Cy = alpha_Cx, alpha_Cy # effective mass parameters in eV*A^2
        self.alpha_Vx, self.alpha_Vy = alpha_Vx, alpha_Vy # effective mass parameters in eV*A^2
        self.d_dist = d_dist # effective parameter for electric field influence in bandgap in Angstrom
        self.E_field = E_field # Electric field in eV/A, seemly opposite with physical direction
        self.N_shell = N_shell # Number of G shells to include

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
        self.dim_H = 4 * self.dim_G # dimension of total Hamiltonian
        
        # Precompute k-independent part of Hamiltonian (Interlayer coupling)
        self.H_const = self._build_constant_hamiltonian()

    def _build_constant_hamiltonian(self):
        """Pre-compute the part of Hamiltonian that doesn't depend on k (Hopping between layers)"""
        H = np.zeros((self.dim_H, self.dim_H), dtype=np.complex128)
        
        # --- Diagonal in G (Interlayer Conduction) ---
        # Tc(0) <-> Bc(2) with coupling gamma_C
        idx = np.arange(self.dim_G) # num of 4x4 blocks
        H[idx * 4 + 0, idx * 4 + 2] = self.gamma_C
        H[idx * 4 + 2, idx * 4 + 0] = self.gamma_C
        
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

        # # [CRITICAL FIX] Add complex phase to break Real-Hamiltonian symmetry
        # # This physically corresponds to a stacking shift r0.
        # # Without this, H is real => Berry Curvature is 0 => Shift Current is 0.
        delta_g = self.G_vectors[rows] - self.G_vectors[cols]
        # r0 = np.array([self.b_lat / 2, 0.0]) # Arbitrary shift approx 0.2 unit cells
        r0 = np.array([0.0, 0.0]) # Arbitrary shift approx 0.2 unit cells
        phase = np.exp(1j * np.dot(delta_g, r0))
        
        H[rows * 4 + 1, cols * 4 + 3] = self.gamma_V * phase
        H[cols * 4 + 3, rows * 4 + 1] = self.gamma_V * np.conj(phase)

        # H[rows * 4 + 1, cols * 4 + 3] = self.gamma_V
        # H[cols * 4 + 3, rows * 4 + 1] = self.gamma_V
        
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
        H_stack = np.tile(self.H_const, (Nk, 1, 1))
        
        # Kinetic Terms (p^2)
        p2_x = kx**2
        p2_y = ky**2
        
        # --- Layer 1 (Top) ---
        # Parameters
        val_Tc = self.e_C0 + u_pot + self.alpha_Cx * p2_x + self.alpha_Cy * p2_y
        val_Tv = self.e_V0 + u_pot + self.alpha_Vx * p2_x + self.alpha_Vy * p2_y
        val_T_mix = self.gamma_ML * kx + self.beta_ML * ky**2
        
        # --- Layer 2 (Bottom) ---
        # Rotated 90 deg: alpha_x <-> alpha_y, p_x -> -p_y
        val_Bc = self.e_C0 - u_pot + self.alpha_Cx * p2_y + self.alpha_Cy * p2_x
        val_Bv = self.e_V0 - u_pot + self.alpha_Vx * p2_y + self.alpha_Vy * p2_x
        val_B_mix = self.gamma_ML * (-ky) + self.beta_ML * kx**2
        
        # Assign to diagonal blocks for all G at once
        # Indices array (Ng,) to broadcast with (Nk, Ng)
        r = np.arange(self.dim_G)
        
        # Diagonals
        H_stack[:, r*4+0, r*4+0] += val_Tc
        H_stack[:, r*4+1, r*4+1] += val_Tv
        H_stack[:, r*4+2, r*4+2] += val_Bc
        H_stack[:, r*4+3, r*4+3] += val_Bv
        
        # Intra-layer mixing (Real)
        H_stack[:, r*4+0, r*4+1] += val_T_mix
        H_stack[:, r*4+1, r*4+0] += val_T_mix
        
        H_stack[:, r*4+2, r*4+3] += val_B_mix
        H_stack[:, r*4+3, r*4+2] += val_B_mix
        
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
        # H_00 (Tc): alpha_Cx * kx^2 -> 2 * alpha_Cx * kx
        vx_stack[:, r*4+0, r*4+0] = 2 * self.alpha_Cx * kx
        # H_11 (Tv): alpha_Vx * kx^2 -> 2 * alpha_Vx * kx
        vx_stack[:, r*4+1, r*4+1] = 2 * self.alpha_Vx * kx
        # H_01/10 (Mix): gamma_ML * kx -> gamma_ML
        vx_stack[:, r*4+0, r*4+1] = self.gamma_ML
        vx_stack[:, r*4+1, r*4+0] = self.gamma_ML
        
        # dH/dky
        # H_00 (Tc): alpha_Cy * ky^2 -> 2 * alpha_Cy * ky
        vy_stack[:, r*4+0, r*4+0] = 2 * self.alpha_Cy * ky
        # H_11 (Tv): alpha_Vy * ky^2 -> 2 * alpha_Vy * ky
        vy_stack[:, r*4+1, r*4+1] = 2 * self.alpha_Vy * ky
        # Mixing term in y
        vy_stack[:, r*4+0, r*4+1] = 2 * self.beta_ML * ky
        vy_stack[:, r*4+1, r*4+0] = 2 * self.beta_ML * ky
        
        # --- Bottom Layer (Rotated: params swapped effectively) ---
        # H_BB uses alpha_Cx*p_y^2 + alpha_Cy*p_x^2
        # where p_x = kx, p_y = ky
        
        # dH/dkx (Only terms with kx)
        # H_22 (Bc): alpha_Cy * kx^2 -> 2 * alpha_Cy * kx
        vx_stack[:, r*4+2, r*4+2] = 2 * self.alpha_Cy * kx
        # H_33 (Bv): alpha_Vy * kx^2 -> 2 * alpha_Vy * kx
        vx_stack[:, r*4+3, r*4+3] = 2 * self.alpha_Vy * kx
        # Mixing
        vx_stack[:, r*4+2, r*4+3] = 2 * self.beta_ML * kx
        vx_stack[:, r*4+3, r*4+2] = 2 * self.beta_ML * kx
        
        # dH/dky (Only terms with ky)
        # H_22 (Bc): alpha_Cx * ky^2 -> 2 * alpha_Cx * ky
        vy_stack[:, r*4+2, r*4+2] = 2 * self.alpha_Cx * ky
        # H_33 (Bv): alpha_Vx * ky^2 -> 2 * alpha_Vx * ky
        vy_stack[:, r*4+3, r*4+3] = 2 * self.alpha_Vx * ky
        # Mixing: gamma_ML * (-ky) -> -gamma_ML
        vy_stack[:, r*4+2, r*4+3] = -self.gamma_ML
        vy_stack[:, r*4+3, r*4+2] = -self.gamma_ML
        
        return vx_stack, vy_stack

    def get_generalized_derivative_matrices(self, k_points):
        """
        w_munu = d^2H / dk_mu dk_nu
        Returns operator matrices w_xx, w_yy, w_xy representing the curvature of the Hamiltonian.
        """
        k_points = np.atleast_2d(k_points)
        Nk = len(k_points)
        
        # Initialize zero matrices of shape (Nk, dim_H, dim_H)
        w_xx = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        w_yy = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        w_xy = np.zeros((Nk, self.dim_H, self.dim_H), dtype=np.complex128)
        
        r = np.arange(self.dim_G)
        
        # --- Top Layer ---
        # Tc(0): alpha_Cx * kx^2 + alpha_Cy * ky^2 => d^2/dx^2 = 2*alpha_Cx
        w_xx[:, r*4+0, r*4+0] = 2 * self.alpha_Cx
        w_yy[:, r*4+0, r*4+0] = 2 * self.alpha_Cy
        
        # Tv(1)
        w_xx[:, r*4+1, r*4+1] = 2 * self.alpha_Vx
        w_yy[:, r*4+1, r*4+1] = 2 * self.alpha_Vy
        
        # --- Bottom Layer (Rotated 90 deg: alpha_x <-> alpha_y) ---
        # Bc(2)
        w_xx[:, r*4+2, r*4+2] = 2 * self.alpha_Cy
        w_yy[:, r*4+2, r*4+2] = 2 * self.alpha_Cx
        
        # Bv(3)
        w_xx[:, r*4+3, r*4+3] = 2 * self.alpha_Vy
        w_yy[:, r*4+3, r*4+3] = 2 * self.alpha_Vx
        
        # --- Mixing Terms ---
        # Top Mixing (0-1): gamma*kx + beta*ky^2 => w_yy = 2*beta
        w_yy[:, r*4+0, r*4+1] = 2 * self.beta_ML
        w_yy[:, r*4+1, r*4+0] = 2 * self.beta_ML
        
        # Bottom Mixing (2-3): -gamma*ky + beta*kx^2 => w_xx = 2*beta
        w_xx[:, r*4+2, r*4+3] = 2 * self.beta_ML
        w_xx[:, r*4+3, r*4+2] = 2 * self.beta_ML
        
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
    plt.plot(k_dist, unfolded_E[:, 0], 'b-', lw=2.5, alpha=0.5, label='Unfolded Bands')
    plt.plot(k_dist, unfolded_E[:, 1:], 'b-', lw=2.5, alpha=0.5)
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
    
    plt.scatter(folded_k, folded_E, s=20, color='red', alpha=0.9, label='Folded Bands',
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
    
    # Vectorize summation over bands
    for i in range(mid_idx):
        for j in range(mid_idx, n_bands):
            # Energy difference
            delta_E = evals[:, j] - evals[:, i] # (Nk,)

            # |\langle c|v_i|v \rangle|^2 / (\Delta E)^2
            #
            # NOTE:
            # The (ΔE)^{-2} factor does NOT belong to the bare transition rate.
            # It appears here because we are constructing \epsilon_2(\omega) (or an absorption-like
            # response), i.e. an electric-field response function.
            #
            # Using \hbar \omega = \Delta E, this factor accounts for the conversion from vector
            # potential A to electric field E, and makes the result equivalent
            # to the length-gauge formulation.
            # Matrix elements for this pair
            M_ij_x = Mx2[:, i, j] / delta_E**2  # (Nk,)
            M_ij_y = My2[:, i, j] / delta_E**2  # (Nk,)
            
            # Add to spectrum
            # Kubo formula component: sum |M|^2 * delta(E - w)
            # We calculate this spectral function S(w) first.
            
            # Broadcast: (N_E, 1) - (1, Nk)
            diff = omegas[:, None] - delta_E[None, :]
            lorentz = (1/np.pi) * eta / (diff**2 + eta**2) # (N_E, Nk)
            
            # Sum over k
            weight_x = np.sum(M_ij_x[None, :] * lorentz, axis=1) # (N_E,)
            weight_y = np.sum(M_ij_y[None, :] * lorentz, axis=1)
            
            sigma_xx += weight_x
            sigma_yy += weight_y

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
                            k_range=0.15, n_k=60,
                            band_window=None, save_prefix="", model=None):
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

    if model is None:
        model = TwistedBPModel(N_shell=N_shell, E_field=E_field)
    
    # 1. Generate Grid
    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    # Diagonalize + velocity (same structure as hBN optical routine)
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack) 
    Nb = evals.shape[1]

    # Velocity / generalized derivative in orbital basis
    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    w_xx, w_yy, w_xy = model.get_generalized_derivative_matrices(k_points)
    
    # Velocity in eigenbasis
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))

    def to_eig(O):
        return U_dag @ O @ U

    # Store component maps
    v_map = {'x': to_eig(vx_orb), 'y': to_eig(vy_orb)}
    w_map = {'xx': to_eig(w_xx), 'yy': to_eig(w_yy), 'xy': to_eig(w_xy), 'yx': to_eig(w_xy)}

    # Band selection
    mid = Nb // 2
    if band_window is None:
        v_idx = list(range(0, mid))
        c_idx = list(range(mid, Nb))
    else:
        v_idx = list(range(band_window[0], band_window[1]+1))
        c_idx = list(range(band_window[2], band_window[3]+1))

    # Spectrum
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

        sigma = np.zeros_like(omegas)
        
        print(f"  Transitions: {len(v_idx)} val x {len(c_idx)} cond, Nk={Nk}")
        
        eps_denom = 1e-5

        for n in v_idx:
            for m in c_idx:
                w_mn = evals[:, m] - evals[:, n]  # (Nk,), > 0 for insulator
                nonzero = w_mn > eps_denom
                
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
                termA[nonzero] = (
                    v_c[nonzero, n, m] * delta_a
                    + v_a[nonzero, n, m] * delta_c
                )
                termA[nonzero] /= (-w_mn[nonzero])
                
                # --- Term B: sum over intermediate states (vectorized over p) ---
                w_np = evals[:, n, None] - evals            # (Nk, Nb)
                w_pm = evals - evals[:, m, None]            # (Nk, Nb)

                valid_p = (np.abs(w_np) > eps_denom) & (np.abs(w_pm) > eps_denom)
                valid_p[:, n] = False
                valid_p[:, m] = False
                valid_p &= nonzero[:, None]

                num1 = v_c[:, n, :] * v_a[:, :, m]
                num2 = v_a[:, n, :] * v_c[:, :, m]

                termB_contrib = np.zeros((Nk, Nb), dtype=np.complex128)
                termB_contrib[valid_p] = (
                    num1[valid_p] / w_pm[valid_p]
                    - num2[valid_p] / w_np[valid_p]
                )
                termB = np.sum(termB_contrib, axis=1)

                # --- Term C: generalized derivative correction ---
                termC = -w_ac[:, n, m]

                # --- Generalized derivative ---
                K_nm = termA + termB + termC # important!
                r_deriv = np.zeros(Nk, dtype=np.complex128)
                r_deriv[nonzero] = K_nm[nonzero] / (-1j * (-w_mn[nonzero]))
                
                # --- Shift current weight ---
                weight = f_nm * np.imag(r_b_mn * r_deriv)
                
                # --- Lorentzian broadening ---
                diff = omegas[:, None] - w_mn[None, :]       # (n_E, Nk)
                lorentz = (1.0/np.pi) * eta / (diff**2 + eta**2)
                
                # Accumulate
                sigma += np.sum(lorentz * weight[None, :], axis=1)

        # Normalize by k-point count
        sigma /= Nk

        # Physical prefactor: sigma has units of Å² so far (from velocity = dH/dk in eV·Å).
        # Full formula: sigma^{abc} = (2*pi*e^2) / (hbar * A_uc) * I(omega)
        # with an extra 1/hbar absorbed into the Lorentzian (delta in 1/eV → 1/(eV·s^{-1}))
        e_charge = 1.602176634e-19   # C
        hbar = 1.054571817e-34       # J·s
        # Calculate Area
        A_uc = 1 / (np.abs(1/b_lat - 1/a_lat))**2 # Å² supercell area based on moiré periodicity
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

def calculate_z_shift_current(N_shell=1, E_field=0.0,
                               E_range=(0.0, 1.0), n_E=400, eta=0.010,
                               k_range=0.15, n_k=60, d_interlayer=5.3,
                               band_window=None):
    r"""
    Calculate out-of-plane (z-direction) shift current: sigma^{z;xx}(omega) and sigma^{z;yy}(omega).

    The z-shift current is:

    \\sigma^{zbb}(\omega) = C * \\sum_{nm}[ f_{nm} \\Im[r^b_{mn} (r^b_{nm})_{;z}] * \\delta(\omega - \omega_{mn})]
                          = C * \\sum_{nm}[ f_{nm} \\(R_{nm}^{b})_{;z}(k) * |r^b_{nm}(k)|^2 * \\delta(\omega - \omega_{mn})]

    For a 2D system where z is NOT periodic, the z-shift vector is simply the
    interlayer charge transfer upon optical excitation:

        (R_{nm}^{b})_{;z}(k) = - <u_n|z|u_n> + <u_m|z|u_m>

    where r^b_{nm} = v^b_{nm} / (i * omega_{nm}) is the interband position matrix element.

    Unlike the in-plane shift current, no covariant derivative (Terms A, B, C) is needed,
    because z is not a crystal momentum direction.

    Parameters
    ----------
    d_interlayer : float
        Physical interlayer distance in Angstrom (default 5.3 A for BP bilayer).
        Top layer orbitals are at z = +d/2, bottom at z = -d/2.
    """
    comp_list = [('z', 'x', 'x'), ('z', 'y', 'y')]

    print(f"Calculating z-shift current (d_interlayer={d_interlayer} A)...")
    model = TwistedBPModel(N_shell=N_shell, E_field=E_field)

    # 1. K-grid
    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    # 2. Diagonalize
    print(f"  Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)
    Nb = evals.shape[1]

    # 3. Velocity matrices in eigenbasis
    print(f"  Calculating velocity matrices...")
    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))

    vx_eig = U_dag @ vx_orb @ U  # (Nk, Nb, Nb)
    vy_eig = U_dag @ vy_orb @ U

    # 4. Build z-position operator and transform to eigenbasis
    #    Top layer (Tc=0, Tv=1): z = +d/2
    #    Bottom layer (Bc=2, Bv=3): z = -d/2
    dim_G = model.dim_G
    dim_H = model.dim_H
    z_op = np.zeros((dim_H, dim_H), dtype=np.float64)
    r = np.arange(dim_G)
    z_op[r*4+0, r*4+0] = +d_interlayer / 2.0
    z_op[r*4+1, r*4+1] = +d_interlayer / 2.0
    z_op[r*4+2, r*4+2] = -d_interlayer / 2.0
    z_op[r*4+3, r*4+3] = -d_interlayer / 2.0

    # Transform: z_eig = U^dag @ z_op @ U  (Nk, Nb, Nb)
    z_eig = U_dag @ z_op @ U
    # Diagonal: <n|z|n> for each band
    z_diag = np.real(np.diagonal(z_eig, axis1=1, axis2=2))  # (Nk, Nb)

    # 5. Band selection
    mid = Nb // 2
    if band_window is None:
        v_idx = np.arange(0, mid)
        c_idx = np.arange(mid, Nb)
    else:
        v_idx = np.arange(band_window[0], band_window[1]+1)
        c_idx = np.arange(band_window[2], band_window[3]+1)

    # 6. Compute z-shift current spectrum
    omegas = np.linspace(E_range[0], E_range[1], n_E)
    eps_denom = 1e-5
    results = {}

    # v_eig[:, n, m] for n in c_idx, m in v_idx
    v_map = {'x': vx_eig, 'y': vy_eig}

    # Energy differences: omega_{nm} = E_n - E_m for n in c_idx, m in v_idx
    # Shape: (Nk, Nv, Nc)
    E_v = evals[:, v_idx]  # (Nk, Nv)
    E_c = evals[:, c_idx]  # (Nk, Nc)
    delta_E = E_c[:, None, :] - E_v[:, :, None]  # (Nk, Nv, Nc)

    # (R_{nm}^{b})_{;z}(k) = - <u_n|z|u_n> + <u_m|z|u_m> for n in c_idx, m in v_idx
    z_v = z_diag[:, v_idx]  # (Nk, Nv)
    z_c = z_diag[:, c_idx]  # (Nk, Nc)
    delta_z = z_v[:, :, None] - z_c[:, None, :]  # (Nk, Nv, Nc)

    print(f"  Transitions: {len(v_idx)} val x {len(c_idx)} cond, Nk={Nk}")

    for comp in comp_list:
        a_dir, b_dir, c_dir = comp
        assert b_dir == c_dir, "z-shift current only for linearly polarized light (b==c)"
        print(f"  Computing sigma^{{{a_dir}{b_dir}{c_dir}}}...")

        v_b = v_map[b_dir]
        # |v^b_{nm}|^2 for n in c_idx, m in v_idx: v_b[:, c, v]
        # v_b[:, n, m] where n in c_idx, m in v_idx
        vb_vc = v_b[:, v_idx, :][:, :, c_idx]  # (Nk, Nv, Nc)
        Mb2 = np.abs(vb_vc)**2  # (Nk, Nv, Nc)

        # |r^b_{nm}|^2 = |v^b_{nm}|^2 / omega_{nm}^2
        # Mask out near-zero energy differences
        valid = delta_E > eps_denom
        r_b_sq = np.zeros_like(Mb2)
        r_b_sq[valid] = Mb2[valid] / delta_E[valid]**2

        # Integrand: f_{nm} * delta_z * |r^b|^2  (f_nm = -1 for c->v)
        integrand = (-1) * delta_z * r_b_sq  # (Nk, Nv, Nc)

        # Flatten pair dimension
        N_pairs = len(v_idx) * len(c_idx)
        integrand_flat = integrand.reshape(Nk, N_pairs)
        dE_flat = delta_E.reshape(Nk, N_pairs)

        # Accumulate spectrum with Lorentzian broadening
        sigma = np.zeros_like(omegas)
        batch_size = max(1, min(N_pairs, max(1, 200_000_000 // (n_E * Nk))))
        for b_start in range(0, N_pairs, batch_size):
            b_end = min(b_start + batch_size, N_pairs)
            dE_batch = dE_flat[:, b_start:b_end]
            int_batch = integrand_flat[:, b_start:b_end]

            diff = omegas[:, None, None] - dE_batch[None, :, :]
            lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
            sigma += np.sum(lorentz * int_batch[None, :, :], axis=(1, 2))

        sigma /= Nk

        # Physical prefactor (same as in-plane shift current)
        e_charge = 1.602176634e-19
        hbar = 1.054571817e-34
        a_lat = model.a_lat
        b_lat = model.b_lat
        A_uc = 1 / (np.abs(1/b_lat - 1/a_lat))**2
        prefactor = (2 * np.pi * e_charge**2) / (hbar * A_uc) * 1E6  # -> muA*A/V^2
        sigma *= prefactor
        results[comp] = sigma

    # 7. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(omegas, results[('z', 'x', 'x')], 'r-', lw=2, label=r'$\sigma^{zxx}(\omega)$')
    plt.plot(omegas, results[('z', 'y', 'y')], 'b--', lw=2, label=r'$\sigma^{zyy}(\omega)$')
    plt.axhline(0, color='k', lw=0.5, ls='--')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel(r'Shift Conductivity ($\mu$A$\cdot$Å/V$^2$)')
    plt.title(f'Out-of-plane Shift Current\nE={E_field} eV/A, d={d_interlayer} A')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)

    fname = f"kp_z_sc.png"
    plt.savefig(fname, dpi=300)
    plt.close()

    print(f"Saved Z-Shift Current Figure: {fname}")
    return omegas, results

if __name__ == "__main__":
    # BP parameters
    a_lat=4.588
    b_lat=3.296
    G_moire = 2 * np.pi * np.abs(1/b_lat - 1/a_lat) # Moiré G vector magnitude for real lattice

    # # single k point test
    # --------------------------------------------
    # model = TwistedBPModel(N_shell=1, E_field=0.0)
    # k_test = np.array([[0.0, 0.0]])
    # H_test = model.get_hamiltonians(k_test)
    # print(H_test[0])
    # print("Eigenvalues at Gamma:", np.linalg.eigvalsh(H_test[0]))
    # plt.imshow(np.abs(H_test[0]), cmap='viridis')
    # plt.colorbar()
    # plt.title("Hamiltonian Magnitude at Gamma")
    # plt.savefig("Hamiltonian_Magnitude_Gamma.png", dpi=200)
    # plt.close()
    
    # Standard 2D Band Structure
    # --------------------------------------------
    # cal_bands(N_shell=1, E_field=0.0, k_fine_steps=200, y_lim=(-0.6,1.4))
    
    # # 3D Band Structure
    # # --------------------------------------------
    # plot_3d_bands(N_shell=1, E_field=0.0, k_range=G_moire/2, n_grid=60, bands_to_plot=4,
    #               view_elev=15, view_azim=45)
                  
    # # Optical Conductivity
    # # --------------------------------------------
    # calculate_optical_conductivity(N_shell=1, E_field=0.0, n_k=160, n_E=500,
    #                                eta=0.010, k_range=G_moire/2, E_range=(0.0, 1.0))
                                   
    # # # Matrix Element Map (VBM -> CBM)
    # # --------------------------------------------
    # plot_transition_matrix_elements(N_shell=1, E_field=0.0, 
    #                                 band_indices=(8, 10),
    #                                 k_range=G_moire/2, n_k=160)
    
    # # # Shift Current Calculation
    # # --------------------------------------------   
    calculate_shift_current(N_shell=1, E_field=0.0, n_k=100, n_E=100,
                            # band_window=(8,9,10,10), 
                            E_range=(0.0, 1.0),
                            save_prefix="", k_range=G_moire/2,)
    
    # Z-direction (out-of-plane) Shift Current
    # --------------------------------------------
    calculate_z_shift_current(N_shell=0, E_field=0.0, n_k=240, n_E=100,
                              E_range=(0.0, 1.0), k_range=G_moire/2,
                              d_interlayer=5.3)