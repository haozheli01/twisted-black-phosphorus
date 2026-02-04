import numpy as np
import matplotlib.pyplot as plt

class TwistedBPModel:
    def __init__(self, N_shell=1, E_field=0.0, 
                 a_lat=3.296, b_lat=4.588,
                 gamma_ML=4.5, gamma_C=0.39, gamma_V=0.07,
                 e_C0=0.69, e_V0=-0.30,
                 alpha_Cx=1.2, alpha_Cy=2.7, alpha_Vx=-5.9, alpha_Vy=-2.0,
                 d_dist=0.54):
        """
        Initialize TwistedBPModel with configurable parameters.
        
        Parameters:
        -----------
        N_shell : int
            Number of shells for plane wave cutoff (default: 1)
        E_field : float
            Electric field in V/Å (default: 0.0)
        a_lat : float
            Lattice constant a in Angstrom (default: 3.296)
        b_lat : float
            Lattice constant b in Angstrom (default: 4.588)
        gamma_ML : float
            Monolayer coupling coefficient in eV (default: 4.5)
        gamma_C : float
            Conduction band interlayer coupling in eV (default: 0.39)
        gamma_V : float
            Valence band interlayer coupling in eV (default: 0.07)
        e_C0 : float
            Conduction band energy offset in eV (default: 0.69)
        e_V0 : float
            Valence band energy offset in eV (default: -0.30)
        alpha_Cx, alpha_Cy : float
            Conduction band effective mass parameters in eV·Å² (default: 1.2, 2.7)
        alpha_Vx, alpha_Vy : float
            Valence band effective mass parameters in eV·Å² (default: -5.9, -2.0)
        d_dist : float
            Electric field distance in Angstrom (default: 0.54)
        """
        
        # lattice constants (Angstrom)
        self.a_lat = a_lat
        self.b_lat = b_lat
        
        # delta_g = 2pi/b - 2pi/a
        self.delta_g_val = 2 * np.pi * np.abs(1/self.b_lat - 1/self.a_lat) # this is the scale of Moiré reciprocal lattice vector
        
        # coupling coefficients (eV)
        self.gamma_ML = gamma_ML
        self.gamma_C = gamma_C
        self.gamma_V = gamma_V
        
        # band parameters (eV)
        # text: epsilon_C0 - epsilon_V0 = 0.99 eV
        self.gap = e_C0 - e_V0
        self.e_C0 = e_C0
        self.e_V0 = e_V0
        
        # effective mass (eV A^2)
        self.alpha_Cx = alpha_Cx
        self.alpha_Cy = alpha_Cy
        self.alpha_Vx = alpha_Vx
        self.alpha_Vy = alpha_Vy
        
        # electric field distance (Angstrom)
        self.d_dist = d_dist
        
        # default electric field (V/Å)
        self.E_field = E_field
        
        # plane wave cutoff (shell)
        self.N_shell = N_shell
        self.G_vectors = self._generate_G_vectors(self.N_shell)
        self.dim_G = len(self.G_vectors)
        self.dim_H = 4 * self.dim_G # 4 orbitals * N_G
        print(f"Initialized TwistedBPModel with {self.dim_G} G-vectors, Hamiltonian dimension {self.dim_H}x{self.dim_H}")
        print(f"  N_shell = {self.N_shell}, E_field = {self.E_field} V/Å")

    def _generate_G_vectors(self, n_shell):
        """generate G vectors up to n_shell"""
        vecs = []
        # Define four coupling directions: (+-dg, 0), (0, +-dg)
        # Here, for simplicity, we generate integer multiples of delta_g on a grid
        for i in range(-n_shell, n_shell + 1):
            for j in range(-n_shell, n_shell + 1):
                vecs.append(np.array([i * self.delta_g_val, j * self.delta_g_val]))
        return np.array(vecs)

    def get_hamiltonian(self, kx, ky, E_field_V_per_A=None):
        """
        construct the Hamiltonian matrix at given kx, ky and electric field
        ref:
        H = | H_TT   H_TB |
            | H_BT   H_BB |
        
        Parameters:
        -----------
        kx, ky : float
            Wave vector components in Å⁻¹
        E_field_V_per_A : float or None
            Electric field in V/Å. If None, uses self.E_field (default)
        """
        
        # Use default E_field if not specified
        if E_field_V_per_A is None:
            E_field_V_per_A = self.E_field
        
        # uT = -uB = eEd/2
        u_pot = E_field_V_per_A * self.d_dist / 2.0
        
        # initialize large matrix (complex)
        H_total = np.zeros((self.dim_H, self.dim_H), dtype=np.complex128)
        
        # auxiliary index function
        def idx(g_index, orb_index):
            # orb_index: 0=Tc, 1=Tv, 2=Bc, 3=Bv
            return g_index * 4 + orb_index

        # --- 1. Fill diagonal blocks (Layer Intra-term) ---
        for i in range(self.dim_G):
            gx, gy = self.G_vectors[i]
            # current total momentum
            k_cur_x = kx + gx
            k_cur_y = ky + gy
            
            # === Top Layer (not rotated) ===
            # p_x -> kx, p_y -> ky (ignore hbar, assume alpha units include it)
            p2_x = k_cur_x**2
            p2_y = k_cur_y**2
            
            # H_TT (1,1) -> Top Conduction
            val_Tc = self.e_C0 + u_pot + self.alpha_Cx * p2_x + self.alpha_Cy * p2_y
            # H_TT (2,2) -> Top Valence
            val_Tv = self.e_V0 + u_pot + self.alpha_Vx * p2_x + self.alpha_Vy * p2_y
            # H_TT (1,2) / (2,1) -> Mixing
            val_T_mix = self.gamma_ML * k_cur_x # gamma * p_x
            
            H_total[idx(i,0), idx(i,0)] = val_Tc
            H_total[idx(i,1), idx(i,1)] = val_Tv
            H_total[idx(i,0), idx(i,1)] = val_T_mix
            H_total[idx(i,1), idx(i,0)] = val_T_mix # Hermitian real
            
            # === Bottom Layer (rotated 90 degrees) ===
            # text: p_hat = (i hbar dy, -i hbar dx) => ( -ky, kx )
            # actually corresponds to coordinate rotation: p_local_x = p_global_y, p_local_y = -p_global_x
            # here simply handle: swap alpha_x and alpha_y for bottom layer kinetic energy
            # mixing term gamma * p_local_x becomes gamma * (-ky)
            
            # H_BB (1,1) -> Bot Conduction
            # alpha_x uses global y k, alpha_y uses global x k
            val_Bc = self.e_C0 - u_pot + self.alpha_Cx * p2_y + self.alpha_Cy * p2_x
            # H_BB (2,2) -> Bot Valence
            val_Bv = self.e_V0 - u_pot + self.alpha_Vx * p2_y + self.alpha_Vy * p2_x
            # H_BB (1,2) -> Mixing
            # gamma * p_local_x => gamma * (-k_cur_y)
            val_B_mix = self.gamma_ML * (-k_cur_y)
            
            H_total[idx(i,2), idx(i,2)] = val_Bc
            H_total[idx(i,3), idx(i,3)] = val_Bv
            H_total[idx(i,2), idx(i,3)] = val_B_mix
            H_total[idx(i,3), idx(i,2)] = val_B_mix # Real
            
            # === Interlayer Coupling (Diagonal in G for Conduction) ===
            # Eq 3: Top-left term is gamma_C (constant)
            # This means it couples the same G component (k-conserving approximation)
            H_total[idx(i,0), idx(i,2)] = self.gamma_C # Tc <-> Bc
            H_total[idx(i,2), idx(i,0)] = self.gamma_C 

        # --- 2. Fill off-diagonal blocks (Moiré Coupling / Hopping) ---
        # Eq 3: sum_{delta_g} e^{i delta_g * r}
        # In reciprocal space, this connects G and G' if G' - G = delta_g
        
        
        for i in range(self.dim_G):
            for j in range(self.dim_G):
                # Calculate difference between two G points
                # Allowed hopping vectors (from Eq 3 sum range: (+-dg, 0), (0, +-dg))
                diff_vec = self.G_vectors[j] - self.G_vectors[i]
                
                # Check if this difference equals some Moiré reciprocal vector delta_g
                # Allow some floating point tolerance
                dist = np.linalg.norm(diff_vec)
                # Normalized check
                if np.abs(dist - self.delta_g_val) < 1e-4:
                    # Further check if it is axial (x or y)
                    if np.abs(diff_vec[0]) < 1e-4 or np.abs(diff_vec[1]) < 1e-4:
                        # This is a valid Moiré coupling connection
                        # Eq 3: Only between Tv (index 1) and Bv (index 3) there is sum e^{igr}
                        # If the phase here is 1 (sum coeff implies 1*e...), then the matrix element is gamma_V
                        
                        # Top Valence (i) <-> Bottom Valence (j)
                        H_total[idx(i,1), idx(j,3)] = self.gamma_V
                        
                        # Ensure Hermiticity H_TB^dagger
                        # H[j, i] = H[i, j]^* (here real)
                        H_total[idx(j,3), idx(i,1)] = self.gamma_V

        return H_total

    def calculate_bands(self, k_path, E_field_V_per_A=None):
        """
        Calculate bands along k_path
        
        Parameters:
        -----------
        k_path : array-like
            Array of k-points, shape (N, 2)
        E_field_V_per_A : float or None
            Electric field in V/Å. If None, uses self.E_field
        """
        bands = []
        for k in k_path:
            H = self.get_hamiltonian(k[0], k[1], E_field_V_per_A)
            evals = np.linalg.eigvalsh(H)
            bands.append(evals)
        return np.array(bands)
    
    def set_E_field(self, E_field):
        """Set the default electric field (V/Å)"""
        self.E_field = E_field
        print(f"Electric field set to {self.E_field} V/Å")
    
    def set_N_shell(self, N_shell):
        """
        Update the number of shells and regenerate G vectors.
        This will change the Hamiltonian dimension.
        """
        self.N_shell = N_shell
        self.G_vectors = self._generate_G_vectors(self.N_shell)
        self.dim_G = len(self.G_vectors)
        self.dim_H = 4 * self.dim_G
        print(f"Updated N_shell = {self.N_shell}, {self.dim_G} G-vectors, Hamiltonian dimension {self.dim_H}x{self.dim_H}")


def run_simulation(N_shell=1, E_field=0.0, k_fine_steps=160, 
                   y_lim=(-1.0, 1.0), save_prefix=""):
    """
    Run the band structure simulation with specified parameters.
    
    Parameters:
    -----------
    N_shell : int
        Number of shells for plane wave cutoff
    E_field : float
        Electric field in V/Å
    k_fine_steps : int
        Number of k-points for scanning
    y_lim : tuple
        Energy range for plotting (E_min, E_max)
    save_prefix : str
        Prefix for output filenames
    
    Returns:
    --------
    model : TwistedBPModel
        The model instance
    unfolded_energies : np.ndarray
        Unfolded band energies
    folded_k_points : list
        Folded k-points
    folded_energies : list
        Folded band energies
    """
    model = TwistedBPModel(N_shell=N_shell, E_field=E_field)
    
    # scale of delta_g
    dg = model.delta_g_val

    # parameters for supercell
    a = model.a_lat
    b = model.b_lat
    L_super = (7 * a + 5 * b) / 2.0
    G_super = dg / 2.0  # for the commensurate 5/7 supercell
    k_boundary = G_super / 2.0

    # Generate k-path: [-dg/2, 0] -> [0, 0] -> [0, dg/2]
    n_seg = k_fine_steps // 2  # points per segment
    
    # Segment 1: [-dg/2, 0] -> [0, 0]
    k_path_1 = np.array([[kx, 0.0] for kx in np.linspace(-dg/2, 0, n_seg, endpoint=False)])
    # Segment 2: [0, 0] -> [0, dg/2]
    # k_path_2 = np.array([[0.0, ky] for ky in np.linspace(0, dg/2, n_seg + 1)])
    k_path_2 = np.array([[ky, 0.0] for ky in np.linspace(0, dg/2, n_seg + 1)])
    
    k_path = np.vstack([k_path_1, k_path_2])
    
    # Calculate cumulative distance along k-path for plotting
    k_dist = [0.0]
    for i in range(1, len(k_path)):
        dk = np.linalg.norm(k_path[i] - k_path[i-1])
        k_dist.append(k_dist[-1] + dk)
    k_dist = np.array(k_dist)
    
    # High symmetry point positions
    k_high_sym_pos = [0.0, k_dist[n_seg], k_dist[-1]]  # X, Gamma, Y
    k_high_sym_labels = [r'$X$', r'$\Gamma$', r'$Y$']
    
    # Folded BZ parameters
    # G_super is the folded BZ reciprocal vector, k_boundary is half of it
    # For folded plot, we need to map each k to the folded BZ and track its position
    
    folded_k_dist = []  # folded k position along the path
    folded_energies = []
    unfolded_energies = []

    # --- 1. Calculate Unfolded Bands (Original Path) ---
    for idx, k in enumerate(k_path):
        kx, ky = k
        H = model.get_hamiltonian(kx, ky)
        evals = np.linalg.eigvalsh(H)
        unfolded_energies.append(evals)

    # --- 2. Calculate Folded Bands (Extended Path) ---
    # We need to scan the full period to capture all folded bands
    # Path X-scan: [-dg/2, dg/2] along kx (ky=0) -> folds to X'-Gamma
    # Path Y-scan: [-dg/2, dg/2] along ky (kx=0) -> folds to Gamma-Y'
    
    # Maintain similar point density
    n_scan = 2 * n_seg + 1
    
    # X-Direction Scan
    k_scan_x = np.linspace(-dg/2, dg/2, n_scan)
    for kx in k_scan_x:
        H = model.get_hamiltonian(kx, 0.0)
        evals = np.linalg.eigvalsh(H)
        
        # Fold kx
        kx_folded = ((kx + k_boundary) % G_super) - k_boundary if G_super > 0 else kx
        
        # Check if it folds into the X'-Gamma segment [-k_boundary, 0]
        # Allow small tolerance
        if -k_boundary - 1e-5 <= kx_folded <= 1e-5:
            # Map to plot coordinate [0, k_boundary]
            # X'(-k_boundary) -> 0.0, Gamma(0) -> k_boundary
            folded_pos = k_boundary + kx_folded
            for e in evals:
                folded_k_dist.append(folded_pos)
                folded_energies.append(e)
                
    # Y-Direction Scan
    k_scan_y = np.linspace(-dg/2, dg/2, n_scan)
    for ky in k_scan_y:
        H = model.get_hamiltonian(0.0, ky)
        evals = np.linalg.eigvalsh(H)
        
        # Fold ky
        ky_folded = ((ky + k_boundary) % G_super) - k_boundary if G_super > 0 else ky
        
        # Check if it folds into the Gamma-Y' segment [0, k_boundary]
        if -1e-5 <= ky_folded <= k_boundary + 1e-5:
            # Map to plot coordinate [k_boundary, 2*k_boundary]
            # Gamma(0) -> k_boundary, Y'(k_boundary) -> 2*k_boundary
            folded_pos = k_boundary + ky_folded
            for e in evals:
                folded_k_dist.append(folded_pos)
                folded_energies.append(e)
    
    unfolded_energies = np.array(unfolded_energies)
    folded_k_dist = np.array(folded_k_dist)
    folded_energies = np.array(folded_energies)
    
    # Generate filename suffix
    suffix = f"_Nshell{N_shell}_E{E_field:.3f}" if save_prefix == "" else f"_{save_prefix}"
    
    # Plot unfolded bands (original k)
    plt.figure(figsize=(8, 8))
    for i in range(len(unfolded_energies[0])):
        plt.plot(k_dist, unfolded_energies[:,i], color='blue', lw=2.5, alpha=0.5, 
                 label='Unfolded Bands' if i==0 else "")
    
    # Add high symmetry point markers
    for pos in k_high_sym_pos:
        plt.axvline(pos, color='gray', linestyle='-', lw=0.5, alpha=0.7)
    plt.xticks(k_high_sym_pos, k_high_sym_labels)
    
    plt.ylim(y_lim)
    plt.xlim(k_dist[0], k_dist[-1])
    plt.xlabel("k-path")
    plt.ylabel("Energy (eV)")
    plt.title(f"N_shell={N_shell}, E_field={E_field} V/Å")
    plt.grid(True, alpha=0.6, axis='y')
    plt.tight_layout()
    plt.savefig(f"TB{suffix}.png", dpi=300)
    plt.close()

    # Plot folded bands
    plt.figure(figsize=(8, 8))
    
    # Folded BZ high symmetry points
    folded_k_high_sym_pos = [0.0, k_boundary, 2*k_boundary]  # X', Gamma, Y'
    folded_k_high_sym_labels = [r"$X'$", r'$\Gamma$', r"$Y'$"]
    
    plt.scatter(folded_k_dist, folded_energies, s=1.0, color='red', alpha=0.75, 
                label='Folded Bands (5/7 Supercell)')
    
    # Add high symmetry point markers
    for pos in folded_k_high_sym_pos:
        plt.axvline(pos, color='gray', linestyle='-', lw=0.5, alpha=0.7)
    plt.xticks(folded_k_high_sym_pos, folded_k_high_sym_labels)
    
    plt.ylim(y_lim)
    plt.xlim(0, 2*k_boundary)
    plt.xlabel("k-path (folded BZ)")
    plt.ylabel("Energy (eV)")
    plt.title(f"N_shell={N_shell}, E_field={E_field} V/Å (Folded)")
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.tight_layout()
    plt.savefig(f"Folded_TB{suffix}.png", dpi=300)
    plt.close()
    
    print(f"Saved: TB{suffix}.png and Folded_TB{suffix}.png")
    
    return model, unfolded_energies, folded_k_dist, folded_energies


if __name__ == "__main__":
    # ============================================================
    # Main parameters - Adjust these to control the simulation
    # ============================================================
    
    # Number of shells for plane wave expansion (controls accuracy & cost)
    N_SHELL = 1  # Try: 1, 2, 3, ...
    
    # Electric field in V/Å
    E_FIELD = -0.4  # Try: 0.0, 0.1, 0.5, 1.0, ...
    
    # Number of k-points for band calculation
    K_FINE_STEPS = 360
    
    # Energy range for plotting
    Y_LIM = (-1.0, 1.0)
    
    # ============================================================
    # Run single simulation
    # ============================================================
    model, unfolded_E, folded_k, folded_E = run_simulation(
        N_shell=N_SHELL,
        E_field=E_FIELD,
        k_fine_steps=K_FINE_STEPS,
        y_lim=Y_LIM
    )
    