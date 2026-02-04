import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

class MonolayerBPModel:
    def __init__(self, 
                 alpha_Cx=1.700, alpha_Cy=2.400, 
                 alpha_Vx=-3.500, alpha_Vy=-0.400,
                 gamma_ML=4.800, beta=3.400,
                 E_c=0.853184, E_v=0.0):
        """
        Initialize k.p model for Monolayer Black Phosphorus.
        Parameters from fit or literature.
        Hamiltonian basis: [Conduction, Valence] (or similar)
        """
        self.alpha_Cx = alpha_Cx
        self.alpha_Cy = alpha_Cy
        self.alpha_Vx = alpha_Vx
        self.alpha_Vy = alpha_Vy
        self.gamma_ML = gamma_ML
        self.beta = beta
        self.E_c = E_c
        self.E_v = E_v

        # Lattice constants
        self.a_lat = 4.588
        self.b_lat = 3.296

    def get_hamiltonians(self, k_points):
        """
        Construct Hamiltonian matrices for a list of k-points.
        H = | Ec + Ak^2      gamma*kx + beta*ky^2 |
            | h.c.           Ev + Bk^2            |
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)
        kx, ky = k_points[:, 0], k_points[:, 1]
        
        H = np.zeros((num_k, 2, 2), dtype=np.complex128)
        
        # Diagonal terms
        diag_C = self.E_c + self.alpha_Cx * kx**2 + self.alpha_Cy * ky**2
        diag_V = self.E_v + self.alpha_Vx * kx**2 + self.alpha_Vy * ky**2
        
        # Off-diagonal
        # From fit_ML.py: gamma_ML * k_x + beta * k_y**2
        off_diag = self.gamma_ML * kx + self.beta * ky**2
        
        H[:, 0, 0] = diag_C
        H[:, 1, 1] = diag_V
        H[:, 0, 1] = off_diag
        H[:, 1, 0] = np.conj(off_diag) # Real if beta, gamma real
        
        return H

    def get_velocity_matrices(self, k_points):
        """
        v_mu = dH/dk_mu
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)
        kx, ky = k_points[:, 0], k_points[:, 1]
        
        vx = np.zeros((num_k, 2, 2), dtype=np.complex128)
        vy = np.zeros((num_k, 2, 2), dtype=np.complex128)
        
        # dH/dkx
        # Diag
        vx[:, 0, 0] = 2 * self.alpha_Cx * kx
        vx[:, 1, 1] = 2 * self.alpha_Vx * kx
        # Off
        vx[:, 0, 1] = self.gamma_ML
        vx[:, 1, 0] = self.gamma_ML
        
        # dH/dky
        # Diag
        vy[:, 0, 0] = 2 * self.alpha_Cy * ky
        vy[:, 1, 1] = 2 * self.alpha_Vy * ky
        # Off
        vy[:, 0, 1] = 2 * self.beta * ky
        vy[:, 1, 0] = 2 * self.beta * ky
        
        return vx, vy
    
    def get_generalized_derivative_matrices(self, k_points):
        """
        w_munu = d^2H / dk_mu dk_nu
        Since H is quadratic, these are constant matrices (independent of k),
        but we return shape (num_k, 2, 2) for broadcasting.
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)
        
        # w_xx
        w_xx = np.zeros((num_k, 2, 2), dtype=np.complex128)
        w_xx[:, 0, 0] = 2 * self.alpha_Cx
        w_xx[:, 1, 1] = 2 * self.alpha_Vx
        
        # w_yy
        w_yy = np.zeros((num_k, 2, 2), dtype=np.complex128)
        w_yy[:, 0, 0] = 2 * self.alpha_Cy
        w_yy[:, 1, 1] = 2 * self.alpha_Vy
        w_yy[:, 0, 1] = 2 * self.beta
        w_yy[:, 1, 0] = 2 * self.beta
        
        # w_xy
        w_xy = np.zeros((num_k, 2, 2), dtype=np.complex128)
        # No cross terms kx*ky in H
        
        return w_xx, w_yy, w_xy

def cal_bands(k_range=None, n_points=100, ylim=(-1, 3)):
    print("Calculating Band Structure...")
    model = MonolayerBPModel()
    
    if k_range is None:
        dg_x = 2 * np.pi / model.a_lat
        dg_y = 2 * np.pi / model.b_lat
        kx_max = dg_x/2
        ky_max = dg_y/2
    else:
        kx_max = k_range
        ky_max = k_range

    # Path: X(-kx_max, 0) -> Gamma(0,0) -> Y(0, ky_max)
    # Segment 1
    k1 = np.zeros((n_points, 2))
    k1[:, 0] = np.linspace(-kx_max, 0, n_points)
    
    # Segment 2
    k2 = np.zeros((n_points, 2))
    k2[:, 1] = np.linspace(0, ky_max, n_points)
    
    k_path = np.vstack([k1, k2])
    
    # Distance
    dists = np.linalg.norm(np.diff(k_path, axis=0), axis=1)
    k_dist = np.concatenate([[0], np.cumsum(dists)])
    
    # Solve
    H = model.get_hamiltonians(k_path)
    evals = np.linalg.eigvalsh(H)
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(k_dist, evals, 'b-', lw=2)
    
    # Ticks
    tick_pos = [0, k_dist[n_points-1], k_dist[-1]]
    tick_lab = ['X', r'$\Gamma$', 'Y']
    for t in tick_pos:
        plt.axvline(t, color='gray', lw=0.5)
        
    plt.xticks(tick_pos, tick_lab)
    plt.ylim(ylim)
    plt.ylabel("Energy (eV)")
    plt.title("Monolayer BP k.p Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ML_Bands.png", dpi=200)
    plt.close()
    print("Saved ML_Bands.png")

def calculate_optical_properties(k_range=0.3, n_k=100, 
                                 E_range=(0.0, 2.0), n_E=500, eta=0.02):
    """
    Calculate Absorption and Shift Current Spectrum.
    """
    print("Calculating Optical Properties (Absorption & Shift Current)...")
    model = MonolayerBPModel()
    
    # 1. Grid
    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)
    
    # 2. Diagonalize
    H = model.get_hamiltonians(k_points) # (Nk, 2, 2)
    evals, evecs = np.linalg.eigh(H)     # (Nk, 2), (N, 2, 2)
    
    # 3. Velocities
    vx, vy = model.get_velocity_matrices(k_points)
    
    # 4. Generalized Derivatives (Curvature of H)
    w_xx, w_yy, w_xy = model.get_generalized_derivative_matrices(k_points)
    
    # 5. Transform to Eigenbasis
    # U_dag @ Op @ U
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))
    
    def to_eig(Op):
        return U_dag @ Op @ U

    v_x_eig = to_eig(vx)
    v_y_eig = to_eig(vy)
    w_xx_eig = to_eig(w_xx)
    w_yy_eig = to_eig(w_yy)
    w_xy_eig = to_eig(w_xy) 
    
    # 6. Spectra Init
    omegas = np.linspace(E_range[0], E_range[1], n_E)
    
    # Absorption
    abs_xx = np.zeros_like(omegas)
    abs_yy = np.zeros_like(omegas)
    
    # Shift Current sigma_munu (response mu, field nu)
    # We calculate sigma_xxx, sigma_xyy, sigma_yxx, sigma_yyy
    shift_xxx = np.zeros_like(omegas)
    shift_xyy = np.zeros_like(omegas) # x current, y field
    shift_yxx = np.zeros_like(omegas) # y current, x field
    shift_yyy = np.zeros_like(omegas)
    
    # 7. Loop over transitions
    # Only 2 bands: 0 -> 1 is the only transition (assuming Gap > 0 and occupied band 0)
    # Check gap
    Eg = evals[:, 1] - evals[:, 0]
    
    # Transition i=0 (Valence), j=1 (Conduction)
    i, j = 0, 1
    
    # Matrix elements
    v_x_mn = v_x_eig[:, i, j] # <0|vx|1>
    v_y_mn = v_y_eig[:, i, j] # <0|vy|1>
    
    # Broadening
    # Calculate for all omegas at once using broadcasting
    # (Nw, Nk)
    delta_w = omegas[:, None] - Eg[None, :]
    lorentz = (1/np.pi) * eta / (delta_w**2 + eta**2)
    
    # --- Absorption ---
    # ~ |v|^2
    M_xx = np.abs(v_x_mn)**2 / Eg**2 
    M_yy = np.abs(v_y_mn)**2 / Eg**2 
    
    # Sum k
    abs_xx = np.sum(M_xx[None, :] * lorentz, axis=1) * omegas
    abs_yy = np.sum(M_yy[None, :] * lorentz, axis=1) * omegas
    
    # --- Shift Current ---
    # Need generalized derivative (v_nm^b)_{;a}
    # Formula: (v_{nm}^b)_{;a} = w_{nm}^{ab} + sum_{l!=n} ... + sum_{l!=m} ...
    # Here n=0, m=1. Indices 0, 1.
    # l != n => l=1. 
    # l != m => l=0.
    
    # Term 1: w_nm
    # Term 2 (l=1): v_{n,1}^a * v_{1,m}^b / (En - E1) = v_{01}^a * v_{11}^b / (E0 - E1)
    # Term 3 (l=0): v_{n,0}^b * v_{0,m}^a / (Em - E0) = v_{00}^b * v_{01}^a / (E1 - E0)
    
    # Energy diff
    E0 = evals[:, 0]
    E1 = evals[:, 1]
    E01 = E0 - E1 # Negative of Gap
    E10 = E1 - E0 # Gap (positive)
    
    # Diagonal velocities (in eigenbasis)
    v_x_00 = v_x_eig[:, 0, 0]
    v_x_11 = v_x_eig[:, 1, 1]
    v_y_00 = v_y_eig[:, 0, 0]
    v_y_11 = v_y_eig[:, 1, 1]
    
    # Off-diagonal velocities already extracted as v_x_mn (0->1)
    # We also need v_{10} = v_{01}^*
    v_x_nm = np.conj(v_x_mn) # <1|vx|0>
    v_y_nm = np.conj(v_y_mn)
    
    # Helper to calc (v_{01}^b)_{;a}
    def calc_gen_deriv(pol_a, pol_b, w_eig):
        # pol_a is index 'x' or 'y' for derivative direction
        # pol_b is index 'x' or 'y' for operator direction
        
        # Select matrices
        if pol_a == 'x': 
            va_eig = v_x_eig
        else: 
            va_eig = v_y_eig
            
        if pol_b == 'x': 
            vb_eig = v_x_eig
            v_b_00 = v_x_00
            v_b_11 = v_x_11
        else: 
            vb_eig = v_y_eig
            v_b_00 = v_y_00
            v_b_11 = v_y_11
        
        # w term
        term_w = w_eig[:, 0, 1] # <0|w|1>
        
        # sum l != 0 (l=1)
        # v_{01}^a * v_{11}^b / (E0 - E1)
        va_01 = va_eig[:, 0, 1]
        term_l1 = (va_01 * v_b_11) / E01
        
        # sum l != 1 (l=0)
        # v_{00}^b * v_{01}^a / (E1 - E0)
        term_l0 = (v_b_00 * va_01) / E10
        
        # Note: va_01 appears in both.
        # Simplifies to va_01 * (v_b_00 - v_b_11) / E10 (since E01 = -E10)
        
        return term_w + term_l1 + term_l0

    # Calculate (v_{01}^b)_{;a} for combinations
    # Responses: 
    # xxx: current x (b=x), field x (nu=x). Need Im[ v_{10}^x * (v_{01}^x)_{;x} ]?
    # Usually formula indices: sigma_mu_nu_nu. Current mu, field nu.
    # Integrand ~ Im[ v_{mn}^nu * (v_{nm}^nu)_{;mu} ]  <-- Check indices carefully
    # Standard: shift current I^a ~ Im[ r_{mn}^b (r_{nm}^b)_{;a} ]
    # Leading to sigma_abb.
    # This ( ... )_{;a} refers to k-derivative wrt 'a'.
    # And r^b is dipole along 'b'.
    # So response is 'a' (Shift current direction), field is 'b' (Polarization).
    # Wait, R vector is geometrical shift.
    # Current J_a = sigma_abb E_b E_b.
    # So we need (v_{01}^b)_{;a}.
    # And multiply by v_{10}^b.
    
    # 1. Sigma_xxx: Current x, Field x. a=x, b=x.
    vd_x_x = calc_gen_deriv('x', 'x', w_xx_eig) # (v_{01}^x)_{;x}
    integrand_xxx = np.imag(v_x_nm * vd_x_x)
    
    # 2. Sigma_xyy: Current x, Field y. a=x, b=y.
    vd_x_y = calc_gen_deriv('x', 'y', w_xx_eig) # (v_{01}^y)_{;x} ?? No w_xx?? 
    # Wait w should be w_ab = d_a d_b H.
    # Here a=x, b=y. w is w_xy.
    # My get_generalized_derivative_matrices return w_xx, w_yy, w_xy
    
    # We need (v^y)_{;x}
    # pol_a = x, pol_b = y. w is w_xy.
    vd_x_y = calc_gen_deriv('x', 'y', w_xy_eig)
    integrand_xyy = np.imag(v_y_nm * vd_x_y)
    
    # 3. Sigma_yxx: Current y, Field x. a=y, b=x.
    vd_y_x = calc_gen_deriv('y', 'x', w_xy_eig)
    integrand_yxx = np.imag(v_x_nm * vd_y_x)
    
    # 4. Sigma_yyy: Current y, Field y. a=y, b=y.
    vd_y_y = calc_gen_deriv('y', 'y', w_yy_eig)
    integrand_yyy = np.imag(v_y_nm * vd_y_y)
    
    # Sum over k and broadcast to omega
    # Shift conductivity ~ 1/omega^2 * sum ...
    
    def integrate_shift(integrand):
        # (Nk,) -> (Nw,)
        return np.sum(integrand[None, :] * lorentz, axis=1) / (omegas**2)  # Check prefactors 
        
    shift_xxx = integrate_shift(integrand_xxx)
    shift_xyy = integrate_shift(integrand_xyy)
    shift_yxx = integrate_shift(integrand_yxx)
    shift_yyy = integrate_shift(integrand_yyy)

    # Normalize densities
    abs_xx /= Nk
    abs_yy /= Nk
    shift_xxx /= Nk
    shift_xyy /= Nk
    shift_yxx /= Nk
    shift_yyy /= Nk
    
    # Plot Absorption
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(omegas, abs_xx, 'r-', label='xx')
    plt.plot(omegas, abs_yy, 'b--', label='yy')
    plt.title('Absorption')
    plt.xlabel('Energy (eV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Shift Current
    plt.subplot(1, 2, 2)
    plt.plot(omegas, shift_xxx, label='xxx (J_x, E_x)')
    plt.plot(omegas, shift_xyy, label='xyy (J_x, E_y)')
    plt.plot(omegas, shift_yxx, label='yxx (J_y, E_x)')
    plt.plot(omegas, shift_yyy, label='yyy (J_y, E_y)')
    plt.title('Shift Current Conductivity (arb. units)')
    plt.xlabel('Energy (eV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ML_Optical.png', dpi=200)
    plt.close()
    print("Saved ML_Optical.png")

def plot_matrix_elements_map(k_range=0.3, n_k=60):
    print("Mapping Matrix Elements...")
    model = MonolayerBPModel()
    
    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    
    vx, vy = model.get_velocity_matrices(k_points)
    H = model.get_hamiltonians(k_points)
    _, evecs = np.linalg.eigh(H)
    
    # 0 -> 1
    u0 = evecs[:, :, 0]
    u1 = evecs[:, :, 1]
    
    # <1|vx|0>
    Mx = np.einsum('ka,kab,kb->k', u1.conj(), vx, u0)
    My = np.einsum('ka,kab,kb->k', u1.conj(), vy, u0)
    
    Zx = (np.abs(Mx)**2).reshape(n_k, n_k)
    Zy = (np.abs(My)**2).reshape(n_k, n_k)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    c1 = ax[0].contourf(KX, KY, Zx, levels=30, cmap='plasma')
    plt.colorbar(c1, ax=ax[0])
    ax[0].set_title(r'$|\langle c | v_x | v \rangle|^2$')
    
    c2 = ax[1].contourf(KX, KY, Zy, levels=30, cmap='plasma')
    plt.colorbar(c2, ax=ax[1])
    ax[1].set_title(r'$|\langle c | v_y | v \rangle|^2$')
    
    plt.tight_layout()
    plt.savefig('ML_MatrixElements.png', dpi=200)
    plt.close()
    print("Saved ML_MatrixElements.png")

if __name__ == "__main__":
    # Lattice constants
    a_lat = 4.588
    b_lat = 3.296
    dg_x = 2 * np.pi / a_lat
    dg_y = 2 * np.pi / b_lat
    dg = np.min([dg_x, dg_y])/2

    cal_bands()
    # plot_matrix_elements_map(k_range= dg)
    # calculate_optical_properties(k_range= dg, E_range=(0.0, 2.5), n_E=500, eta=0.01, n_k=400)
