import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

class TwistedBPModel:
    def __init__(self, 
                 N_top=1, N_bottom=1, twist_angle=0.0):
        """
        Initialize TwistedBPModel with configurable parameters.
        Lattice parameters are obtained from my DFT calculations for monolayer BP.
        The TB model is based on Rudenko et al. PHYSICAL REVIEW B 92, 085419 (2015) and PHYSICAL REVIEW B 93, 199906(E) (2016).

        Here, we use an 2x2 effective model to describe the low-energy physics of a naturally-stacking multilayer BP system,
        following PHYSICAL REVIEW B 96, 155427 (2017):
            H = H0 + H2 + cos(N * pi / (N + 1)) * H3
        where H0, H2 are intralayer terms, and H3 is the interlayer coupling term, N is the number of layers in the stack.

        For a specific multilayer BP, above Hamiltonian can be readily diagonalized:
            H = [[E_cond, 0], [0, E_val]].
        And the bottom layer can be twisted by an angle, which modifies the intralayer terms H0, H2 and interlayer coupling H3 accordingly.

        For the coupling between the top and bottom layers, we use an effective coupling strength to capture the main physics of the interface,
        following C Sevik et al. 2D Mater. 4 (2017) 035025.

        The total hamiltonian is:
        H = [[H_top, V_sub], [V_sub^dagger, H_bot]]
        """
        # monolayer BP parameters
        self.b_lat=4.588 # armchair direction
        self.a_lat=3.296 # zig-zag direction
        self.N_top = N_top
        self.N_bottom = N_bottom
        self.twist_angle = twist_angle

        # effective interface coupling strength (in eV)
        self.coupling = 0.095

        # tight-binding parameters
        self.a1 = 2.22
        self.a2 = 2.24
        self.alpha1 = 96.5 * np.pi / 180
        self.alpha2 = 101.9 * np.pi / 180
        self.beta = 72.0 * np.pi / 180
        # intralayer hoppings (in eV)
        self.t1=-1.486
        self.t2=3.729
        self.t3=-0.252
        self.t4=-0.071
        self.t5= 0.019 # should be +0.019
        self.t6=0.186
        self.t7=-0.063
        self.t8=0.101
        self.t9=-0.042
        self.t10=0.073
        # interlayer hoppings (in eV)
        self.t1p=0.524
        self.t2p=0.180
        self.t3p=-0.123
        self.t4p=-0.168
        # self.t5p=0.005 # we dont use this here

    def basic_block(self, k_points, twist_angle):
        """
        Compute the basic Hamiltonian for given k-points.
        """
        k_points = np.atleast_2d(k_points)
        kx_tmp, ky_tmp = k_points[:, 0], k_points[:, 1]
        kx = kx_tmp * np.cos(twist_angle) - ky_tmp * np.sin(twist_angle)
        ky = kx_tmp * np.sin(twist_angle) + ky_tmp * np.cos(twist_angle)

        # Intralayer terms
        tAA = 2 * self.t3 * np.cos(2 * self.a1 * np.sin(self.alpha1/2) * kx ) \
            + 2 * self.t7 * np.cos((2 * self.a1 * np.sin(self.alpha1/2) + 2 * self.alpha2 * np.cos(self.beta)) * ky) \
            + 4 * self.t10 * np.cos(2 * self.a1 * np.sin(self.alpha1/2) * kx) * np.cos((2 * self.a1 * np.sin(self.alpha1/2) + 2 * self.alpha2 * np.cos(self.beta)) * ky)
        
        tAB = 2 * self.t1 * np.cos(self.a1 * np.sin(self.alpha1/2) * kx ) * np.exp(-1j * (self.a1 * np.cos(self.alpha1/2) * ky)) \
            + 2 * self.t4 * np.cos(self.a1 * np.sin(self.alpha1/2) * kx ) * np.exp(1j * (self.a1 * np.cos(self.alpha1/2) + 2 * self.a2 * np.cos(self.beta)) * ky) \
            + 2 * self.t8 * np.cos(3 * self.a1 * np.sin(self.alpha1/2) * kx) * np.exp(-1j * (self.a1 * np.cos(self.alpha1/2) * ky))
        
        tAC = self.t2 * np.exp(1j * self.a2 * np.cos(self.beta) * ky) \
            + self.t6 * np.exp(-1j * (2 * self.a1 * np.cos(self.alpha1/2) + self.a2 * np.cos(self.beta)) * ky) \
            + 2 * self.t9 * np.cos(2 * self.a1 * np.sin(self.alpha1/2) * kx) * np.exp(-1j * (self.a2 * np.cos(self.beta) + 2 * self.a1 * np.cos(self.alpha1/2)) * ky)
        
        tAD = 4 * self.t5 * np.cos(self.a1 * np.sin(self.alpha1/2) * kx) * np.cos((self.a1 * np.cos(self.alpha1/2) + self.a2 * np.cos(self.beta)) * ky)

        # Interlayer terms
        tADp = (4 * self.t3p * np.cos(2 * self.a1 * np.sin(self.alpha1/2) * kx ) + 2* self.t2p) \
            * np.cos((self.a1 * np.sin(self.alpha1/2) + self.alpha2 * np.cos(self.beta)) * ky)
        
        tACp = (2 * self.t1p * np.exp(1j * self.a2 * np.cos(self.beta) * ky) + 2 * self.t4p * np.exp(-1j * (2 * self.a1 * np.sin(self.alpha1/2) + self.a2 * np.cos(self.beta)) * ky)) \
            * np.cos(2 * self.a1 * np.sin(self.alpha1/2) * kx)
    

        return tAA, tAB, tAC, tAD, tADp, tACp

    def get_hamiltonians(self, k_points):
        """
        Compute the Hamiltonian in sublattice basis.
        Each layer block is H = [[a, z], [z*, a]]
        Interlayer coupling V_sub is a constant matrix.
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)

        # Top layer (untwisted)
        prefactor = np.cos(np.pi * self.N_top / (self.N_top + 1))
        tAA, tAB, tAC, tAD, tADp, tACp = self.basic_block(k_points, twist_angle=0.0)
        a_t = tAA + tAD + prefactor * tADp
        z_t = tAB + tAC + prefactor * tACp
        ham_top = np.zeros((num_k, 2, 2), dtype=np.complex128)
        ham_top[:, 0, 0] = a_t
        ham_top[:, 1, 1] = a_t
        ham_top[:, 0, 1] = z_t
        ham_top[:, 1, 0] = np.conj(z_t)

        if self.N_bottom == 0:
            return ham_top

        # Bottom layer (twisted)
        prefactor = np.cos(np.pi * self.N_bottom / (self.N_bottom + 1))
        tAA, tAB, tAC, tAD, tADp, tACp = self.basic_block(k_points, twist_angle=self.twist_angle)
        a_b = tAA + tAD + prefactor * tADp
        z_b = tAB + tAC + prefactor * tACp
        ham_bot = np.zeros((num_k, 2, 2), dtype=np.complex128)
        ham_bot[:, 0, 0] = a_b
        ham_bot[:, 1, 1] = a_b
        ham_bot[:, 0, 1] = z_b
        ham_bot[:, 1, 0] = np.conj(z_b)

        ham = np.zeros((num_k, 4, 4), dtype=np.complex128)
        ham[:, :2, :2] = ham_top
        ham[:, 2:4, 2:4] = ham_bot

        # interlayer coupling
        ham_inter = np.zeros((num_k, 2, 2), dtype=np.complex128)
        ham_inter[:, 0, 0] = self.coupling
        ham_inter[:, 0, 1] = self.coupling
        ham_inter[:, 1, 0] = self.coupling
        ham_inter[:, 1, 1] = self.coupling

        ham[:, :2, 2:4] = ham_inter
        ham[:, 2:4, :2] = ham_inter.conj().transpose((0, 2, 1))

        return ham

    def basic_block_velocity(self, k_points, twist_angle):
        """
        Compute dH0/dk, dH2/dk, dH3/dk w.r.t. global (unrotated) k coordinates.
        Returns (dH0_x, dH0_y, dH2_x, dH2_y, dH3_x, dH3_y), each (Nk, 2, 2).
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)
        c = np.cos(twist_angle)
        s = np.sin(twist_angle)
        kx_tmp, ky_tmp = k_points[:, 0], k_points[:, 1]
        kx = kx_tmp * c - ky_tmp * s
        ky = kx_tmp * s + ky_tmp * c

        # Geometric projections
        sx = self.a1 * np.sin(self.alpha1 / 2)
        cx = self.a1 * np.cos(self.alpha1 / 2)
        cy = self.a2 * np.cos(self.beta)
        ay = self.alpha2 * np.cos(self.beta)

        p1 = 2 * sx;  p2 = sx;  p3 = cx;  p4 = cx + 2 * cy
        p5 = cy;  p6 = 2 * cx + cy;  p8 = cx + cy;  p9 = 3 * sx
        q1 = 2 * sx + 2 * ay;  q2 = sx + ay;  p10 = 2 * sx + cy

        # === Local derivatives of each element ===
        # dtAA/dkx, dtAA/dky
        dtAA_x = -2 * self.t3 * p1 * np.sin(p1 * kx) \
                 - 4 * self.t10 * p1 * np.sin(p1 * kx) * np.cos(q1 * ky)
        dtAA_y = -2 * self.t7 * q1 * np.sin(q1 * ky) \
                 - 4 * self.t10 * np.cos(p1 * kx) * q1 * np.sin(q1 * ky)

        # dtAB/dkx, dtAB/dky
        e1 = np.exp(-1j * p3 * ky);  e4 = np.exp(1j * p4 * ky)
        dtAB_x = -2 * self.t1 * p2 * np.sin(p2 * kx) * e1 \
                 - 2 * self.t4 * p2 * np.sin(p2 * kx) * e4 \
                 - 2 * self.t8 * p9 * np.sin(p9 * kx) * e1
        dtAB_y = 2 * self.t1 * np.cos(p2 * kx) * (-1j * p3) * e1 \
               + 2 * self.t4 * np.cos(p2 * kx) * (1j * p4) * e4 \
               + 2 * self.t8 * np.cos(p9 * kx) * (-1j * p3) * e1

        # dtAC/dkx, dtAC/dky
        e5 = np.exp(1j * p5 * ky);  e6 = np.exp(-1j * p6 * ky)
        dtAC_x = -2 * self.t9 * p1 * np.sin(p1 * kx) * e6
        dtAC_y = self.t2 * (1j * p5) * e5 \
               + self.t6 * (-1j * p6) * e6 \
               + 2 * self.t9 * np.cos(p1 * kx) * (-1j * p6) * e6

        # dtAD/dkx, dtAD/dky
        dtAD_x = -4 * self.t5 * p2 * np.sin(p2 * kx) * np.cos(p8 * ky)
        dtAD_y = -4 * self.t5 * np.cos(p2 * kx) * p8 * np.sin(p8 * ky)

        # dtADp/dkx, dtADp/dky
        dtADp_x = -4 * self.t3p * p1 * np.sin(p1 * kx) * np.cos(q2 * ky)
        dtADp_y = -(4 * self.t3p * np.cos(p1 * kx) + 2 * self.t2p) * q2 * np.sin(q2 * ky)

        # dtACp/dkx, dtACp/dky
        f_ky = 2 * self.t1p * e5 + 2 * self.t4p * np.exp(-1j * p10 * ky)
        df_ky = 2 * self.t1p * (1j * p5) * e5 + 2 * self.t4p * (-1j * p10) * np.exp(-1j * p10 * ky)
        dtACp_x = f_ky * (-p1 * np.sin(p1 * kx))
        dtACp_y = df_ky * np.cos(p1 * kx)

        # === Transform to global k via chain rule ===
        def to_global(dx, dy):
            return dx * c + dy * s, -dx * s + dy * c

        gtAA_x,  gtAA_y  = to_global(dtAA_x,  dtAA_y)
        gtAB_x,  gtAB_y  = to_global(dtAB_x,  dtAB_y)
        gtAC_x,  gtAC_y  = to_global(dtAC_x,  dtAC_y)
        gtAD_x,  gtAD_y  = to_global(dtAD_x,  dtAD_y)
        gtADp_x, gtADp_y = to_global(dtADp_x, dtADp_y)
        gtACp_x, gtACp_y = to_global(dtACp_x, dtACp_y)

        return (gtAA_x, gtAA_y, 
                gtAB_x, gtAB_y, 
                gtAC_x, gtAC_y, 
                gtAD_x, gtAD_y, 
                gtADp_x, gtADp_y, 
                gtACp_x, gtACp_y)

    def basic_block_curvature(self, k_points, twist_angle):
        """
        Compute d^2H/dk_mu dk_nu for basic blocks w.r.t. global k coordinates.
        Returns (d2H0_xx, d2H0_yy, d2H0_xy,
                 d2H2_xx, d2H2_yy, d2H2_xy,
                 d2H3_xx, d2H3_yy, d2H3_xy), each (Nk, 2, 2).
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)
        c = np.cos(twist_angle)
        s = np.sin(twist_angle)
        kx_tmp, ky_tmp = k_points[:, 0], k_points[:, 1]
        kx = kx_tmp * c - ky_tmp * s
        ky = kx_tmp * s + ky_tmp * c

        # Geometric projections (same as velocity)
        sx = self.a1 * np.sin(self.alpha1 / 2)
        cx = self.a1 * np.cos(self.alpha1 / 2)
        cy = self.a2 * np.cos(self.beta)
        ay = self.alpha2 * np.cos(self.beta)

        p1 = 2 * sx;  p2 = sx;  p3 = cx;  p4 = cx + 2 * cy
        p5 = cy;  p6 = 2 * cx + cy;  p8 = cx + cy;  p9 = 3 * sx
        q1 = 2 * sx + 2 * ay;  q2 = sx + ay;  p10 = 2 * sx + cy

        # Precompute trig/exp factors
        cos_p1kx = np.cos(p1 * kx);  sin_p1kx = np.sin(p1 * kx)
        cos_p2kx = np.cos(p2 * kx);  sin_p2kx = np.sin(p2 * kx)
        cos_p9kx = np.cos(p9 * kx);  sin_p9kx = np.sin(p9 * kx)
        cos_q1ky = np.cos(q1 * ky);  sin_q1ky = np.sin(q1 * ky)
        cos_p8ky = np.cos(p8 * ky);  sin_p8ky = np.sin(p8 * ky)
        cos_q2ky = np.cos(q2 * ky);  sin_q2ky = np.sin(q2 * ky)
        e1 = np.exp(-1j * p3 * ky);  e4 = np.exp(1j * p4 * ky)
        e5 = np.exp(1j * p5 * ky);   e6 = np.exp(-1j * p6 * ky)
        e10 = np.exp(-1j * p10 * ky)

        # === Local second derivatives ===
        # tAA
        d2tAA_xx = -2 * self.t3 * p1**2 * cos_p1kx - 4 * self.t10 * p1**2 * cos_p1kx * cos_q1ky
        d2tAA_yy = -2 * self.t7 * q1**2 * cos_q1ky - 4 * self.t10 * cos_p1kx * q1**2 * cos_q1ky
        d2tAA_xy = 4 * self.t10 * p1 * sin_p1kx * q1 * sin_q1ky

        # tAB
        d2tAB_xx = -2 * self.t1 * p2**2 * cos_p2kx * e1 \
                   - 2 * self.t4 * p2**2 * cos_p2kx * e4 \
                   - 2 * self.t8 * p9**2 * cos_p9kx * e1
        d2tAB_yy = -2 * self.t1 * p3**2 * cos_p2kx * e1 \
                   - 2 * self.t4 * p4**2 * cos_p2kx * e4 \
                   - 2 * self.t8 * p3**2 * cos_p9kx * e1
        d2tAB_xy = 2 * self.t1 * p2 * p3 * 1j * sin_p2kx * e1 \
                 - 2 * self.t4 * p2 * p4 * 1j * sin_p2kx * e4 \
                 + 2 * self.t8 * p9 * p3 * 1j * sin_p9kx * e1

        # tAC
        d2tAC_xx = -2 * self.t9 * p1**2 * cos_p1kx * e6
        d2tAC_yy = -self.t2 * p5**2 * e5 - self.t6 * p6**2 * e6 \
                   - 2 * self.t9 * p6**2 * cos_p1kx * e6
        d2tAC_xy = 2 * self.t9 * p1 * p6 * 1j * sin_p1kx * e6

        # tAD
        d2tAD_xx = -4 * self.t5 * p2**2 * cos_p2kx * cos_p8ky
        d2tAD_yy = -4 * self.t5 * p8**2 * cos_p2kx * cos_p8ky
        d2tAD_xy = 4 * self.t5 * p2 * p8 * sin_p2kx * sin_p8ky

        # tADp
        d2tADp_xx = -4 * self.t3p * p1**2 * cos_p1kx * cos_q2ky
        d2tADp_yy = -(4 * self.t3p * cos_p1kx + 2 * self.t2p) * q2**2 * cos_q2ky
        d2tADp_xy = 4 * self.t3p * p1 * q2 * sin_p1kx * sin_q2ky

        # tACp
        f_ky = 2 * self.t1p * e5 + 2 * self.t4p * e10
        df_ky = 2 * self.t1p * (1j * p5) * e5 + 2 * self.t4p * (-1j * p10) * e10
        d2f_ky = -2 * self.t1p * p5**2 * e5 - 2 * self.t4p * p10**2 * e10

        d2tACp_xx = f_ky * (-p1**2 * cos_p1kx)
        d2tACp_yy = d2f_ky * cos_p1kx
        d2tACp_xy = df_ky * (-p1 * sin_p1kx)

        # === Transform to global via rotation ===
        c2 = c**2;  s2 = s**2;  cs = c * s;  c2ms2 = c2 - s2

        def to_global_2nd(dxx, dyy, dxy):
            gxx = c2 * dxx + 2 * cs * dxy + s2 * dyy
            gyy = s2 * dxx - 2 * cs * dxy + c2 * dyy
            gxy = -cs * dxx + c2ms2 * dxy + cs * dyy
            return gxx, gyy, gxy

        gtAA_xx,  gtAA_yy,  gtAA_xy  = to_global_2nd(d2tAA_xx,  d2tAA_yy,  d2tAA_xy)
        gtAB_xx,  gtAB_yy,  gtAB_xy  = to_global_2nd(d2tAB_xx,  d2tAB_yy,  d2tAB_xy)
        gtAC_xx,  gtAC_yy,  gtAC_xy  = to_global_2nd(d2tAC_xx,  d2tAC_yy,  d2tAC_xy)
        gtAD_xx,  gtAD_yy,  gtAD_xy  = to_global_2nd(d2tAD_xx,  d2tAD_yy,  d2tAD_xy)
        gtADp_xx, gtADp_yy, gtADp_xy = to_global_2nd(d2tADp_xx, d2tADp_yy, d2tADp_xy)
        gtACp_xx, gtACp_yy, gtACp_xy = to_global_2nd(d2tACp_xx, d2tACp_yy, d2tACp_xy)


        return (gtAA_xx, gtAA_yy, gtAA_xy,
                gtAB_xx, gtAB_yy, gtAB_xy,
                gtAC_xx, gtAC_yy, gtAC_xy,
                gtAD_xx, gtAD_yy, gtAD_xy,
                gtADp_xx, gtADp_yy, gtADp_xy,
                gtACp_xx, gtACp_yy, gtACp_xy)

    def _layer_velocity(self, k_points, twist_angle, prefactor):
        """
        Compute d/dk_mu of the 2x2 block for one layer (sublattice basis).

        H_layer = [[a, z], [z*, a]]

        d/dk_mu H = [[da/dk_mu,  dz/dk_mu],
                     [dz*/dk_mu, da/dk_mu]]

        Returns vx, vy each (Nk, 2, 2).
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)

        # First derivatives of each hopping element
        (dAA_x, dAA_y, dAB_x, dAB_y, dAC_x, dAC_y,
         dAD_x, dAD_y, dADp_x, dADp_y, dACp_x, dACp_y) = \
            self.basic_block_velocity(k_points, twist_angle)

        da_x = dAA_x + dAD_x + prefactor * dADp_x
        da_y = dAA_y + dAD_y + prefactor * dADp_y
        dz_x = dAB_x + dAC_x + prefactor * dACp_x
        dz_y = dAB_y + dAC_y + prefactor * dACp_y

        vx = np.zeros((num_k, 2, 2), dtype=np.complex128)
        vy = np.zeros((num_k, 2, 2), dtype=np.complex128)

        vx[:, 0, 0] = da_x
        vx[:, 1, 1] = da_x
        vx[:, 0, 1] = dz_x
        vx[:, 1, 0] = np.conj(dz_x)

        vy[:, 0, 0] = da_y
        vy[:, 1, 1] = da_y
        vy[:, 0, 1] = dz_y
        vy[:, 1, 0] = np.conj(dz_y)

        return vx, vy

    def get_velocity_matrices(self, k_points):
        """
        Calculate velocity matrices vx, vy for the full Hamiltonian.
        v = dH/dk (units: eV * Angstrom)
        Returns vx, vy each (Nk, dim_H, dim_H).
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)

        pf_t = np.cos(np.pi * self.N_top / (self.N_top + 1))
        vx_t, vy_t = self._layer_velocity(k_points, 0.0, pf_t)

        if self.N_bottom == 0:
            return vx_t, vy_t

        pf_b = np.cos(np.pi * self.N_bottom / (self.N_bottom + 1))
        vx_b, vy_b = self._layer_velocity(k_points, self.twist_angle, pf_b)

        # Interlayer coupling derivatives are all zeros since we assume a constant coupling strength

        vx = np.zeros((num_k, 4, 4), dtype=np.complex128)
        vy = np.zeros((num_k, 4, 4), dtype=np.complex128)
        vx[:, :2, :2] = vx_t
        vx[:, 2:4, 2:4] = vx_b
        vy[:, :2, :2] = vy_t
        vy[:, 2:4, 2:4] = vy_b

        return vx, vy

    def _layer_curvature(self, k_points, twist_angle, prefactor):
        """
        Compute d^2/dk_mu dk_nu of the 2x2 block for one layer (sublattice basis).

        d^2H/dk_mu dk_nu = [[d^2a,  d^2z ],
                            [d^2z*, d^2a ]]

        Returns w_xx, w_yy, w_xy each (Nk, 2, 2).
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)

        # Second derivatives of each hopping element
        (d2AA_xx, d2AA_yy, d2AA_xy,
         d2AB_xx, d2AB_yy, d2AB_xy,
         d2AC_xx, d2AC_yy, d2AC_xy,
         d2AD_xx, d2AD_yy, d2AD_xy,
         d2ADp_xx, d2ADp_yy, d2ADp_xy,
         d2ACp_xx, d2ACp_yy, d2ACp_xy) = \
            self.basic_block_curvature(k_points, twist_angle)

        results = []
        for (d2AA, d2AD, d2ADp, d2AB, d2AC, d2ACp) in [
            (d2AA_xx, d2AD_xx, d2ADp_xx, d2AB_xx, d2AC_xx, d2ACp_xx),  # xx
            (d2AA_yy, d2AD_yy, d2ADp_yy, d2AB_yy, d2AC_yy, d2ACp_yy),  # yy
            (d2AA_xy, d2AD_xy, d2ADp_xy, d2AB_xy, d2AC_xy, d2ACp_xy),  # xy
        ]:
            d2a = d2AA + d2AD + prefactor * d2ADp
            d2z = d2AB + d2AC + prefactor * d2ACp

            w = np.zeros((num_k, 2, 2), dtype=np.complex128)
            w[:, 0, 0] = d2a
            w[:, 1, 1] = d2a
            w[:, 0, 1] = d2z
            w[:, 1, 0] = np.conj(d2z)
            results.append(w)

        return results[0], results[1], results[2]

    def get_generalized_derivative_matrices(self, k_points):
        """
        w_munu = d^2H / dk_mu dk_nu for the full Hamiltonian.
        Returns w_xx, w_yy, w_xy each (Nk, dim_H, dim_H).
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)

        pf_t = np.cos(np.pi * self.N_top / (self.N_top + 1))
        wxx_t, wyy_t, wxy_t = self._layer_curvature(k_points, 0.0, pf_t)

        if self.N_bottom == 0:
            return wxx_t, wyy_t, wxy_t

        pf_b = np.cos(np.pi * self.N_bottom / (self.N_bottom + 1))
        wxx_b, wyy_b, wxy_b = self._layer_curvature(k_points, self.twist_angle, pf_b)

        # Interlayer coupling second derivatives are also zeros since we assume a constant coupling strength

        results = []
        for w_t, w_b in [(wxx_t, wxx_b),
                               (wyy_t, wyy_b),
                               (wxy_t, wxy_b)]:
            w = np.zeros((num_k, 4, 4), dtype=np.complex128)
            w[:, :2, :2] = w_t
            w[:, 2:4, 2:4] = w_b
            results.append(w)

        return results[0], results[1], results[2]

# ============================================================
# Standalone analysis functions
# ============================================================

def cal_bands(N_top=4, N_bottom=4, twist_angle=0.0,
              k_fine_steps=360, y_lim=(-2, 1.5), save_prefix=""):
    """Calculate and plot band structure along X -> Gamma -> Y, with folded BZ."""
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom,
                           twist_angle=twist_angle)

    b_lat = model.b_lat
    a_lat = model.a_lat
    dg = 2 * np.pi * np.abs(1/b_lat - 1/a_lat)
    G_super = dg / 2.0
    k_boundary = G_super / 2.0

    # Path: X -> Gamma -> Y
    n_seg = k_fine_steps // 2
    path_1 = np.zeros((n_seg, 2))
    path_1[:, 0] = np.linspace(-dg/2, 0, n_seg, endpoint=False)
    path_2 = np.zeros((n_seg + 1, 2))
    path_2[:, 1] = np.linspace(0, dg/2, n_seg + 1)
    k_path = np.vstack([path_1, path_2])

    dists = np.linalg.norm(np.diff(k_path, axis=0), axis=1)
    k_dist = np.concatenate([[0], np.cumsum(dists)])
    sym_pos = [0.0, k_dist[n_seg], k_dist[-1]]
    sym_labels = [r'$X$', r'$\Gamma$', r'$Y$']

    # Unfolded bands
    print(f"Calculating unfolded bands ({len(k_path)} k-points)...")
    H = model.get_hamiltonians(k_path)
    unfolded_E = np.linalg.eigvalsh(H)

    # Folded bands via 2D folding into commensurate supercell BZ
    # G_super = dg/2, k_boundary = G_super/2 = dg/4
    # Each segment has 4 moiré paths that fold onto it.
    # X'->Gamma (kx: -k_boundary->0, ky=0):
    #   (-k_boundary,0)->(0,0),  (k_boundary,0)->(G_super,0),
    #   (-k_boundary,G_super)->(0,G_super),  (k_boundary,G_super)->(G_super,G_super)
    # Gamma->Y' (kx=0, ky: 0->k_boundary):
    #   (0,0)->(0,k_boundary),  (G_super,0)->(G_super,k_boundary),
    #   (0,-G_super)->(0,-k_boundary),  (G_super,-G_super)->(G_super,-k_boundary)
    print(f"Calculating folded bands (2D folding with 4 paths per segment)...")

    n_dim = H.shape[-1]

    # --- X'->Gamma segment ---
    kc_XG = np.column_stack([np.linspace(-k_boundary, 0, n_seg, endpoint=False),
                              np.zeros(n_seg)])
    shifts_XG = np.array([[0, 0], [G_super, 0], [0, G_super], [G_super, G_super]])
    E_XG = np.zeros((n_seg, len(shifts_XG) * n_dim))
    for s_idx, shift in enumerate(shifts_XG):
        k_moire = kc_XG + shift[np.newaxis, :]
        eigs = np.linalg.eigvalsh(model.get_hamiltonians(k_moire))
        E_XG[:, s_idx*n_dim:(s_idx+1)*n_dim] = eigs

    # --- Gamma->Y' segment ---
    kc_GY = np.column_stack([np.zeros(n_seg + 1),
                              np.linspace(0, k_boundary, n_seg + 1)])
    shifts_GY = np.array([[0, 0], [G_super, 0], [0, -G_super], [G_super, -G_super]])
    E_GY = np.zeros((n_seg + 1, len(shifts_GY) * n_dim))
    for s_idx, shift in enumerate(shifts_GY):
        k_moire = kc_GY + shift[np.newaxis, :]
        eigs = np.linalg.eigvalsh(model.get_hamiltonians(k_moire))
        E_GY[:, s_idx*n_dim:(s_idx+1)*n_dim] = eigs

    # Combine and sort
    kc_path = np.vstack([kc_XG, kc_GY])
    dk_folded = np.linalg.norm(np.diff(kc_path, axis=0), axis=1)
    folded_k_dist = np.concatenate([[0], np.cumsum(dk_folded)])
    folded_energies = np.sort(np.vstack([E_XG, E_GY]), axis=1)

    n = H.shape[-1]
    print(f"  Band gap ≈ {np.min(unfolded_E[:, n//2] - unfolded_E[:, n//2 - 1]):.4f} eV")

    plot_2D_bands(k_dist, unfolded_E, folded_k_dist, folded_energies,
                  sym_pos, sym_labels, k_boundary, y_lim,
                  suffix=f"N{N_top}_{N_bottom}_tw{np.degrees(twist_angle):.0f}{save_prefix}",
                  folded_is_structured=True)
    return model


def plot_2D_bands(k_dist, unfolded_E, folded_k, folded_E,
                  sym_pos, sym_labels, k_boundary, y_lim, suffix="",
                  folded_is_structured=False):
    """Plot unfolded and folded band structures."""
    nbnd = unfolded_E.shape[-1]
    vbm = np.max(unfolded_E[:, :nbnd//2])

    # Unfolded plot
    plt.figure(figsize=(8, 8))
    plt.plot(k_dist, unfolded_E[:, 0] - vbm, 'b-', lw=2.5, alpha=0.5, label='Unfolded Bands')
    plt.plot(k_dist, unfolded_E[:, 1:] - vbm, 'b-', lw=2.5, alpha=0.5)
    for pos in sym_pos:
        plt.axvline(pos, c='gray', ls='-', lw=0.5)
    plt.xticks(sym_pos, sym_labels)
    plt.ylim(y_lim)
    plt.xlim(k_dist[0], k_dist[-1])
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"EM_unfolded_{suffix}.png", dpi=200)
    plt.close()

    # Folded plot
    if len(folded_k) > 0:
        plt.figure(figsize=(8, 8))
        folded_sym_pos = [0.0, k_boundary, 2*k_boundary]
        folded_sym_labels = [r"$X'$", r'$\Gamma$', r"$Y'$"]
        if folded_is_structured:
            # Structured: folded_k is (n_k,), folded_E is (n_k, n_bands_folded)
            plt.plot(folded_k, folded_E[:, 0] - vbm, 'r-', lw=2.5, alpha=0.5, label='Folded Bands')
            plt.plot(folded_k, folded_E[:, 1:] - vbm, 'r-', lw=2.5, alpha=0.5)
        else:
            # Legacy scatter format
            plt.scatter(folded_k, folded_E - vbm, s=20, color='red', alpha=0.9,
                        label='Folded Bands', facecolors='white', edgecolors='red',
                        linewidths=0.5, zorder=0)
        for pos in folded_sym_pos:
            plt.axvline(pos, c='gray', ls='-', lw=0.5)
        plt.xticks(folded_sym_pos, folded_sym_labels)
        plt.ylim(y_lim)
        plt.xlim(0, 2*k_boundary)
        plt.ylabel("Energy (eV)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"EM_folded_{suffix}.png", dpi=200)
        plt.close()

    print(f"Figures saved with suffix {suffix}")


def plot_3d_bands(N_top=1, N_bottom=1, twist_angle=0.0,
                  k_range=0.2, n_grid=40, bands_to_plot=4,
                  view_elev=30, view_azim=45, save_prefix=""):
    """Plot 3D band structure surface on a 2D k-grid."""
    print(f"Generating 3D band plot with range [{-k_range}, {k_range}]...")
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    kx = np.linspace(-k_range, k_range, n_grid)
    ky = np.linspace(-k_range, k_range, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])

    evals = np.linalg.eigvalsh(model.get_hamiltonians(k_points))
    n_bands = evals.shape[1]
    E_grid = evals.reshape(n_grid, n_grid, n_bands)

    mid_idx = n_bands // 2
    start_band = max(0, mid_idx - bands_to_plot // 2)
    end_band = min(n_bands, mid_idx + bands_to_plot // 2)

    fig = plt.figure(figsize=(8.0, 8.0))
    ax = fig.add_subplot(111, projection='3d')
    for b in range(start_band, end_band):
        Z = E_grid[:, :, b]
        cmap = 'viridis' if b < mid_idx else 'plasma'
        ax.plot_surface(KX, KY, Z, cmap=cmap, alpha=0.9,
                        edgecolor='none', antialiased=False)

    ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    ax.set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    ax.set_zlabel('Energy (eV)')
    ax.set_title('3D Band Structure')
    ax.set_box_aspect((1, 1, 1.5))
    ax.view_init(elev=view_elev, azim=view_azim)
    plt.tight_layout()
    fname = f"EM_3D{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved 3D plot: {fname}")


def calculate_optical_conductivity(N_top=1, N_bottom=1, twist_angle=0.0,
                                   E_range=(0.0, 1.0), n_E=500, eta=0.010,
                                   k_range=0.15, n_k=60, save_prefix=""):
    r"""
    Calculate optical absorption spectrum via Kubo-Greenwood formula.

        \epsilon_2^{ii}(\omega) \propto
        \sum_{v,c} \int_{BZ} d^2k
        |\langle c,k | v_i | v,k \rangle|^2 / (E_{c,k} - E_{v,k})^2
        \cdot \delta(E_{c,k} - E_{v,k} - \hbar \omega)

    Absorbance is proportional to \epsilon_2 * \omega, so we plot that as the final spectrum.
    """
    print(f"Calculating optical conductivity spectrum...")
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    print(f"  Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)

    print(f"  Calculating velocity matrices...")
    vx_stack, vy_stack = model.get_velocity_matrices(k_points)

    U_dag = np.conj(np.transpose(evecs, (0, 2, 1)))
    vx_eig = U_dag @ vx_stack @ evecs
    vy_eig = U_dag @ vy_stack @ evecs

    Mx2 = np.abs(vx_eig)**2
    My2 = np.abs(vy_eig)**2

    omegas = np.linspace(E_range[0], E_range[1], n_E)
    sigma_xx = np.zeros_like(omegas)
    sigma_yy = np.zeros_like(omegas)

    n_bands = evals.shape[1]
    mid_idx = n_bands // 2

    Mx2_vc = Mx2[:, :mid_idx, mid_idx:]
    My2_vc = My2[:, :mid_idx, mid_idx:]

    delta_E_all = evals[:, mid_idx:, None] - evals[:, None, :mid_idx]
    delta_E_all = delta_E_all.transpose(0, 2, 1)

    M_weighted_x = Mx2_vc / delta_E_all**2
    M_weighted_y = My2_vc / delta_E_all**2

    N_pairs = mid_idx * (n_bands - mid_idx)
    M_flat_x = M_weighted_x.reshape(Nk, N_pairs)
    M_flat_y = M_weighted_y.reshape(Nk, N_pairs)
    dE_flat = delta_E_all.reshape(Nk, N_pairs)

    print(f"  Summing transitions...")
    batch_size = max(1, min(N_pairs, max(1, 200_000_000 // (n_E * Nk))))
    for b_start in range(0, N_pairs, batch_size):
        b_end = min(b_start + batch_size, N_pairs)
        dE_batch = dE_flat[:, b_start:b_end]
        Mx_batch = M_flat_x[:, b_start:b_end]
        My_batch = M_flat_y[:, b_start:b_end]
        diff = omegas[:, None, None] - dE_batch[None, :, :]
        lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
        sigma_xx += np.sum(lorentz * Mx_batch[None, :, :], axis=(1, 2))
        sigma_yy += np.sum(lorentz * My_batch[None, :, :], axis=(1, 2))

    sigma_xx /= Nk
    sigma_yy /= Nk
    absorption_xx = sigma_xx * omegas
    absorption_yy = sigma_yy * omegas

    plt.figure(figsize=(8, 6))
    plt.plot(omegas, absorption_xx, 'r-', label=r'x-polarized', lw=2)
    plt.plot(omegas, absorption_yy, 'b--', label=r'y-polarized', lw=2)
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Optical Absorption (a.u.)')
    plt.title(rf'$\eta$={eta*1000:.1f} meV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)

    fname = f"EM_absorption{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved Optical Absorption Spectrum: {fname}")


def plot_transition_matrix_elements(N_top=1, N_bottom=1, twist_angle=0.0,
                                    band_indices=None,
                                    k_range=0.15, n_k=60, save_prefix=""):
    r"""
    Plot |<j|v|i>|^2 in k-space as contour plots.
    """
    print(f"Calculating transition matrix elements map...")
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    dim_H = 4 if N_bottom > 0 else 2
    if band_indices is None:
        mid = dim_H // 2
        band_i = mid - 1  # VBM
        band_j = mid       # CBM
    else:
        band_i, band_j = band_indices

    print(f"  Mapping transition: Band {band_i} -> Band {band_j}")

    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])

    _, evecs = np.linalg.eigh(model.get_hamiltonians(k_points))
    vx_stack, vy_stack = model.get_velocity_matrices(k_points)

    u_i = evecs[:, :, band_i]
    u_j = evecs[:, :, band_j]

    M_x = np.einsum('ka,kab,kb->k', u_j.conj(), vx_stack, u_i)
    M_y = np.einsum('ka,kab,kb->k', u_j.conj(), vy_stack, u_i)

    Z_x = (np.abs(M_x)**2).reshape(n_k, n_k)
    Z_y = (np.abs(M_y)**2).reshape(n_k, n_k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    zs = np.concatenate([Z_x.flatten(), Z_y.flatten()])
    vmax = np.percentile(zs, 99)

    c1 = axes[0].contourf(KX, KY, Z_x, levels=40, cmap='plasma', vmin=0, vmax=vmax)
    axes[0].set_title(r'$|\langle \psi_f | v_x | \psi_i \rangle|^2$')
    axes[0].set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    axes[0].set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    axes[0].set_aspect('equal')
    plt.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(KX, KY, Z_y, levels=40, cmap='plasma', vmin=0, vmax=vmax)
    axes[1].set_title(r'$|\langle \psi_f | v_y | \psi_i \rangle|^2$')
    axes[1].set_xlabel(r'$k_x$ ($\AA^{-1}$)')
    axes[1].set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    axes[1].set_aspect('equal')
    plt.colorbar(c2, ax=axes[1])

    plt.suptitle(f"Transition Matrix Elements: Band {band_i} -> {band_j}")
    plt.tight_layout()
    fname = f"EM_M_B{band_i}-{band_j}{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved Matrix Element Map: {fname}")


def calculate_z_shift_current(N_top=1, N_bottom=1, twist_angle=0.0,
                               E_range=(0.0, 1.0), n_E=400, eta=0.010,
                               k_range=0.15, n_k=60, layerthickness=5.2,
                               band_window=None, save_prefix=""):
    r"""
    Calculate out-of-plane (z-direction) shift current: sigma^{z;xx}(omega) and sigma^{z;yy}(omega).

    The z-shift current is:

    \\sigma^{zbb}(\omega) = C * \\sum_{nm}[ f_{nm} \\Im[r^b_{mn} (r^b_{nm})_{;z}] * \\delta(\omega - \omega_{nm})]
                          = C * \\sum_{nm}[ f_{nm} \\(R_{nm}^{b})_{;z}(k) * |r^b_{nm}(k)|^2 * \\delta(\omega - \omega_{nm})]

    For a 2D system where z is NOT periodic, the z-shift vector is simply the
    interlayer charge transfer upon optical excitation:

        (R_{nm}^{b})_{;z}(k) = - <u_n|z|u_n> + <u_m|z|u_m>

    where r^b_{nm} = v^b_{nm} / (i * omega_{nm}) is the interband position matrix element.

    Unlike the in-plane shift current, no covariant derivative (Terms A, B, C) is needed,
    because z is not a crystal momentum direction.

    Parameters
    ----------
    layerthickness : float
        Physical layer thickness in Angstrom (default 5.2 A).
        Top layer orbitals are at z = +d/2 * N_top, bottom at z = -d/2 * N_bottom.
    """
    comp_list = [('z', 'x', 'x'), ('z', 'y', 'y')]
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

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
    vx_eig = U_dag @ vx_orb @ U
    vy_eig = U_dag @ vy_orb @ U

    # z-position operator: top layer +d/2, bottom layer -d/2
    dim_H = Nb
    z_op = np.zeros((dim_H, dim_H), dtype=np.float64)
    z_op[0, 0] = +layerthickness * N_top / 2.0  # top conduction
    z_op[1, 1] = +layerthickness * N_top / 2.0  # top valence
    if dim_H == 4:
        z_op[2, 2] = -layerthickness * N_bottom / 2.0  # bottom conduction
        z_op[3, 3] = -layerthickness * N_bottom / 2.0  # bottom valence

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
        V_uc = 1 / (np.abs(1/b_lat - 1/a_lat))**2 * layerthickness * (N_top + N_bottom)  # Effective unit cell volume in Å^3
        prefactor = (2 * np.pi * e_charge**2) / (hbar * V_uc) * 1E6
        sigma *= prefactor
        results[comp] = sigma

    # 7. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(omegas, results[('z', 'x', 'x')], 'r-', lw=2, label=r'$\sigma^{zxx}(\omega)$')
    plt.plot(omegas, results[('z', 'y', 'y')], 'b--', lw=2, label=r'$\sigma^{zyy}(\omega)$')
    plt.axhline(0, color='k', lw=0.5, ls='--')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel(r'Shift Conductivity ($\mu$A/V$^2$)')
    plt.title(f'Out-of-plane Shift Current')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)
    fname = f"EM_z_sc{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved Z-Shift Current Figure: {fname}")
    return omegas, results


def calculate_shift_current(N_top=1, N_bottom=1, twist_angle=0.0,
                            E_range=(0.0, 1.0), n_E=400, eta=0.010,
                            k_range=0.15, n_k=60, band_window=None,
                            save_prefix=""):
    r"""
    Shift current sigma^{abb}(omega) via gauge-invariant Sum-Over-States method.

    Ref: Phys. Rev. B 61, 5337 (2000)

        \sigma^{abb}(\omega) = C * \sum_{nm}[ f_{nm} \Im[r^b_{mn} (r^b_{nm})_{;a}]
                               * \delta(\omega - \omega_{nm})]

    where:
        r^b_{mn} = v^b_{mn} / (i \omega_{mn})
        (r^b_{nm})_{;a} = (-1/i\omega_{nm}) [ term_A/\omega_{nm} + term_B + term_C ]
        term_A = v^b_{nm} \delta^a_{nm} + v^a_{nm} \delta^b_{nm}
        term_B = \sum_{p\neq n,m} [v^b_{np} v^a_{pm}/\omega_{pm} - v^a_{np} v^b_{pm}/\omega_{np}]
        term_C = - w^{ab}_{nm}  (generalized derivative of velocity)
    """
    comp_list = [('x', 'x', 'x'), ('x', 'y', 'y'), ('y', 'x', 'x'), ('y', 'y', 'y')]
    plt.figure(figsize=(8, 6))

    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

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
              f"(n_k={n_k}, eta={eta*1000:.0f} meV)...")

        v_a = v_map[a_dir]
        v_b = v_map[b_dir]
        v_c = v_map[c_dir]
        w_ac = w_map[a_dir + c_dir]

        sigma = np.zeros_like(omegas)
        print(f"  Transitions: {len(v_idx)} val x {len(c_idx)} cond, Nk={Nk}")

        # Because of \delta(\omega - \omega_{nm}), here we assum n is cond and m is val

        for n in c_idx:
            f_n = 1.0 if n < mid else 0.0

            # Precompute n-row velocity slices once per n
            v_c_n_row = v_c[:, n, :]  # (Nk, Nb)
            v_a_n_row = v_a[:, n, :]  # (Nk, Nb)
            v_a_nn = v_a[:, n, n]     # (Nk,)
            v_c_nn = v_c[:, n, n]     # (Nk,)
            w_n_all = w_all[:, n, :]  # (Nk, Nb)  w_np = E_n - E_p

            for m in v_idx:
                f_m = 1.0 if m < mid else 0.0
                f_nm = f_n - f_m
                if f_nm == 0.0:
                    continue

                w_nm = evals[:, n] - evals[:, m]  # (Nk,)
                nonzero = w_nm > eps_denom

                r_b_mn = np.zeros(Nk, dtype=np.complex128)
                r_b_mn[nonzero] = v_b[nonzero, m, n] / (-1j * w_nm[nonzero])

                # Term A
                termA = np.zeros(Nk, dtype=np.complex128)
                delta_a = v_a_nn[nonzero] - v_a[nonzero, m, m]
                delta_c = v_c_nn[nonzero] - v_c[nonzero, m, m]
                termA[nonzero] = (v_c[nonzero, n, m] * delta_a
                                + v_a[nonzero, n, m] * delta_c) / (w_nm[nonzero])

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
                r_deriv[nonzero] = K_nm[nonzero] / (-1j * (w_nm[nonzero]))

                weight = f_nm * np.imag(r_b_mn * r_deriv)
                diff = omegas[:, None] - w_nm[None, :]
                lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
                sigma += np.sum(lorentz * weight[None, :], axis=1)

        sigma /= Nk

        e_charge = 1.602176634e-19
        hbar = 1.054571817e-34
        a_lat = model.a_lat
        b_lat = model.b_lat
        A_uc = 1 / (np.abs(1/b_lat - 1/a_lat))**2
        prefactor = (2 * np.pi * e_charge**2) / (hbar * A_uc) * 1E6
        sigma *= prefactor
        results[comp] = sigma

    plt.plot(omegas, results[('x', 'x', 'x')], 'r-', lw=2, label=r'$\sigma^{xxx}(\omega)$')
    plt.plot(omegas, results[('y', 'y', 'y')], 'b--', lw=2, label=r'$\sigma^{yyy}(\omega)$')
    plt.plot(omegas, results[('y', 'x', 'x')], 'g-', lw=2, label=r'$\sigma^{yxx}(\omega)$')
    plt.plot(omegas, results[('x', 'y', 'y')], 'm--', lw=2, label=r'$\sigma^{xyy}(\omega)$')
    plt.axhline(0, color='k', lw=0.5, ls='--')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel(r'Shift Conductivity ($\mu$A$\cdot$Å/V$^2$)')
    plt.title('Shift Current Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)
    fname = f"EM_sc{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved Shift Current Figure: {fname}")
    return omegas, results


def plot_bandgap_scaling(N_top=1, N_bottom=1, twist_angle=0.0,
                         E_g=2.1, gamma_c = 0.58, gamma_v = -0.32,):
    """
    Plot how the gap changes with number of bottom layer changing.
    """
    # Get the bandgap of my model
    gap_level = []
    for N_bot in range(N_bottom[0], N_bottom[1]):
        model = TwistedBPModel(N_top=N_top, N_bottom=N_bot, twist_angle=twist_angle)
        k_points = np.array([[0.0, 0.0]])
        evals, _ = np.linalg.eigh(model.get_hamiltonians(k_points))
        mid = evals.shape[1] // 2
        E_gap = evals[0, mid] - evals[0, mid-1]
        gap_level.append(E_gap)
        print(f"N_bottom={N_bot}: Bandgap = {E_gap:.3f} eV")

    # Get the analytic levels, ref: Huang et al., Science 386, 526–531 (2024)
    level_list = np.linspace(N_bottom[0], N_bottom[1]+3, 500)
    Y_bright = E_g - 2 * gamma_c * np.cos(np.pi/(level_list + N_top + 1)) + 2 * gamma_v * np.cos(np.pi/(N_top + 1))
    X_bright = E_g - 2 * gamma_c * np.cos(np.pi/(level_list + N_top + 1)) + 2 * gamma_v * np.cos(np.pi/(level_list + 1))
    gap_level = np.array(gap_level)

    # Plotting
    plt.figure(figsize=(5, 5))
    plt.plot(level_list, X_bright, label='X-bright (analytic)', color='red', ls='--')
    plt.plot(level_list, Y_bright, label='Y-bright (analytic)', color='blue', ls='--')
    plt.scatter(range(N_bottom[0], N_bottom[1]), gap_level, label='model', color='red', marker='o')
    # plt.tight_layout()
    plt.ylim(0.2,1.6)
    plt.legend()
    plt.xlabel('N_bottom')
    plt.ylabel('Bandgap (eV)')
    fname = f"gap.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved: {fname}")

# =====================================================================
#  Bethe-Salpeter Equation (BSE) — Excitonic Z-Shift Current
# =====================================================================

def keldysh_potential(q, kappa=2.5, r0=5.0, N_top=1, N_bottom=1):
    """
    2D Keldysh-screened Coulomb potential V(q) in eV·Å^2.

    V(q) = 2π * e^2 / κ(q * (1 + r₀*q)) (in CGS unit)
         = 14.3996 * 2π / κ(q * (1 + r₀*q)) (in SI unit, with e^2/(4πε₀) = 14.3996 eV·Å)

    Parameters
    ----------
    q : array_like
        Momentum transfer magnitudes (1/Å).
    kappa : float
        Effective dielectric constant of environment (default 2.5 for hBN).
    r0 : float
        2D polarizability screening length in Å (default 5.0 for BP).
    N_top, N_bottom : int
        Number of top and bottom layers.
    """
    q = np.asarray(q, dtype=np.float64)
    V = np.zeros_like(q)
    mask = q > 1e-12
    V[mask] = 14.3996 * 2 * np.pi / (kappa *(q[mask] * (1 + r0 * (N_top + N_bottom) * q[mask])))
    return V


def build_bse_hamiltonian(evals, evecs, k_points, v_idx, c_idx, A_uc, kappa=2.5, r0=5.0, N_top=1, N_bottom=1):
    """
    Build BSE Hamiltonian in the electron-hole product basis |v,c,k⟩.

    H_BSE = diag(E_c - E_v) - K_d

    where the direct (attractive) kernel is:
        K_d(vck, v'c'k') = V(|k-k'|)/(Nk*A_uc) * ⟨u_c(k)|u_c'(k')⟩ * ⟨u_v'(k')|u_v(k)⟩

    Parameters
    ----------
    evals : ndarray, shape (Nk, Nb)
    evecs : ndarray, shape (Nk, Nb, Nb)  — columns are eigenstates
    k_points : ndarray, shape (Nk, 2)
    v_idx, c_idx : array of band indices
    A_uc : float
        Unit cell area in Å².
    kappa, r0 : Keldysh parameters
    """
    Nk = len(k_points)
    Nv = len(v_idx)
    Nc = len(c_idx)
    dim_bse = Nv * Nc * Nk

    print(f"  Building BSE Hamiltonian: {Nv}v x {Nc}c x {Nk}k = {dim_bse} basis states")
    print(f"    Memory estimate: {dim_bse**2 * 16 / 1e9:.2f} GB")

    # --- Diagonal: QP transition energies ---
    E_v = evals[:, v_idx]  # (Nk, Nv)
    E_c = evals[:, c_idx]  # (Nk, Nc)
    delta_E = (E_c[:, None, :] - E_v[:, :, None])  # (Nk, Nv, Nc)
    diag_vals = delta_E.reshape(Nk * Nv * Nc)

    H_bse = np.diag(diag_vals).astype(np.complex128)

    # --- Coulomb kernel ---
    dk = k_points[:, None, :] - k_points[None, :, :]  # (Nk, Nk, 2)
    q_mag = np.linalg.norm(dk, axis=2)  # (Nk, Nk)
    Vq = keldysh_potential(q_mag, kappa=kappa, r0=r0, N_top=N_top, N_bottom=N_bottom) / (Nk * A_uc)  # (Nk, Nk), BZ integration weight

    # Wavefunction overlaps
    U_c = evecs[:, :, c_idx]  # (Nk, dim_H, Nc)
    U_v = evecs[:, :, v_idx]  # (Nk, dim_H, Nv)

    print(f"    Computing conduction overlaps...")
    overlap_cc = np.einsum('kai,laj->klij', U_c.conj(), U_c)  # (Nk, Nk, Nc, Nc)

    print(f"    Computing valence overlaps...")
    overlap_vv = np.einsum('lai,kaj->lkij', U_v.conj(), U_v)  # (Nk, Nk, Nv, Nv)

    print(f"    Assembling kernel (vectorized over k-pairs)...")
    for ik in range(Nk):
        Vq_row = Vq[ik, :]
        ov_cc = overlap_cc[ik, :, :, :]  # (Nk', Nc, Nc')
        ov_vv = overlap_vv[:, ik, :, :]  # (Nk', Nv', Nv)

        # kernel_full[v, c, k', v', c'] = Vq[k'] * ov_cc[k', c, c'] * ov_vv[k', v', v]
        kernel_full = np.einsum('q,qij,qpv->viqpj', Vq_row, ov_cc, ov_vv)

        row_start = ik * Nv * Nc
        row_end = row_start + Nv * Nc
        kernel_2d = kernel_full.reshape(Nv * Nc, Nk * Nv * Nc)
        H_bse[row_start:row_end, :] -= kernel_2d

    # Diagnostics
    n_nan = np.count_nonzero(np.isnan(H_bse))
    n_inf = np.count_nonzero(np.isinf(H_bse))
    print(f"    NaN count: {n_nan}, Inf count: {n_inf}")
    print(f"    H_bse max |element|: {np.nanmax(np.abs(H_bse)):.6e}")
    print(f"    H_bse diagonal range: [{np.min(np.real(np.diag(H_bse))):.4f}, {np.max(np.real(np.diag(H_bse))):.4f}] eV")

    herm_err = np.max(np.abs(H_bse - H_bse.conj().T))
    print(f"    Hermiticity error: {herm_err:.2e}")
    if herm_err > 1e-8:
        print(f"    WARNING: Large Hermiticity error! Symmetrizing...")
        H_bse = 0.5 * (H_bse + H_bse.conj().T)

    return H_bse


def calculate_bse_z_shift_current(N_top=1, N_bottom=1, twist_angle=0.0,
                                   E_range=(0.0, 1.0), n_E=400, eta=0.010,
                                   k_range=0.15, n_k_bse=30,
                                   n_val=2, n_cond=2,
                                   thickness=5.2,
                                   kappa=2.5, r0=5.0,
                                   plot_ipa_comparison=True,
                                   band_window=None, save_prefix=""):
    r"""
    Excitonic z-shift current via the Bethe-Salpeter equation (BSE).

    Solves the BSE to obtain exciton wavefunctions, then computes:

        sigma^{z;bb}(ω) = C * Σ_S Re[d^{b*}_S * g^{bz}_S] * δ(Ω_S - ω)

    where:
        d^b_S = Σ_{vck} A^S_{vck} r^b_{cv}(k)    (exciton optical dipole)
        g^{bz}_S = Σ_{vck} A^S_{vck} Δz(vck) r^b_{cv}(k)  (z-weighted dipole)

    Parameters
    ----------
    n_k_bse : int
        k-grid per direction for BSE (total Nk = n_k_bse²).
    n_val, n_cond : int
        Number of valence/conduction bands in BSE active space.
    thickness : float
        Thickness in Å.
    kappa : float
        Effective dielectric constant of environment.
    r0 : float
        2D polarizability screening length in Å.
    plot_ipa_comparison : bool
        If True, overlay IPA z-shift current for comparison.
    """
    comp_list = [('z', 'x', 'x'), ('z', 'y', 'y')]

    print("=" * 60)
    print("BSE Excitonic Z-Shift Current Calculation (Effective Model)")
    print("=" * 60)
    print(f"  Grid: {n_k_bse}x{n_k_bse} = {n_k_bse**2} k-points")
    print(f"  Active space: {n_val}v x {n_cond}c")
    print(f"  BSE dimension: {n_val * n_cond * n_k_bse**2}")
    print(f"  Keldysh params: kappa={kappa}, r0={r0} Å")
    print(f"  thickness={thickness} Å")

    # 1. Model & k-grid
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    kx = np.linspace(-k_range, k_range, n_k_bse)
    ky = np.linspace(-k_range, k_range, n_k_bse)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    # 2. Single-particle solve
    print(f"\n[1] Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)
    Nb = evals.shape[1]

    # 3. Select active bands near the gap
    mid = Nb // 2
    if band_window is not None:
        v_idx = np.arange(band_window[0], band_window[1] + 1)
        c_idx = np.arange(band_window[2], band_window[3] + 1)
    else:
        v_idx = np.arange(mid - n_val, mid)
        c_idx = np.arange(mid, mid + n_cond)

    Nv = len(v_idx)
    Nc = len(c_idx)
    print(f"  Bands: valence {v_idx}, conduction {c_idx}")
    print(f"  QP gap range: {np.min(evals[:, c_idx[0]] - evals[:, v_idx[-1]]):.4f} - "
          f"{np.max(evals[:, c_idx[-1]] - evals[:, v_idx[0]]):.4f} eV")

    # 4. Velocity matrices in eigenbasis
    print(f"\n[2] Computing velocity and z-operator...")
    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))

    vx_eig = U_dag @ vx_orb @ U  # (Nk, Nb, Nb)
    vy_eig = U_dag @ vy_orb @ U

    # 5. z-operator diagonal in eigenbasis
    dim_H = Nb
    z_op = np.zeros((dim_H, dim_H), dtype=np.float64)
    z_op[0, 0] = +thickness * N_top / 2.0  # top sub-A
    z_op[1, 1] = +thickness * N_top / 2.0  # top sub-B
    if dim_H == 4:
        z_op[2, 2] = -thickness * N_bottom / 2.0  # bottom sub-A
        z_op[3, 3] = -thickness * N_bottom / 2.0  # bottom sub-B

    z_eig = U_dag @ z_op @ U
    z_diag = np.real(np.diagonal(z_eig, axis1=1, axis2=2))  # (Nk, Nb)

    # delta_z[k, v, c] = z_vv(k) - z_cc(k) for each (v, c) pair
    z_v = z_diag[:, v_idx]  # (Nk, Nv)
    z_c = z_diag[:, c_idx]  # (Nk, Nc)
    delta_z = z_v[:, :, None] - z_c[:, None, :]  # (Nk, Nv, Nc)

    # 6. Interband position matrix elements r^b_{cv}(k) = v^b_{cv} / (i * omega_{cv})
    E_v_arr = evals[:, v_idx]  # (Nk, Nv)
    E_c_arr = evals[:, c_idx]  # (Nk, Nc)
    dE = E_c_arr[:, None, :] - E_v_arr[:, :, None]  # (Nk, Nv, Nc)

    eps_denom = 1e-5
    v_eig_map = {'x': vx_eig, 'y': vy_eig}

    # r^b_{cv}(k) for each direction — shape (Nk, Nv, Nc)
    r_b = {}
    for b_dir in ['x', 'y']:
        v_b = v_eig_map[b_dir]
        vb_cv = v_b[:, c_idx, :][:, :, v_idx]  # (Nk, Nc, Nv)
        vb_cv = np.transpose(vb_cv, (0, 2, 1))  # (Nk, Nv, Nc)
        rb = np.zeros_like(vb_cv)
        valid = np.abs(dE) > eps_denom
        rb[valid] = vb_cv[valid] / (1j * dE[valid])
        r_b[b_dir] = rb  # (Nk, Nv, Nc)

    # 7. Build & diagonalize BSE Hamiltonian
    # Moiré unit cell area
    a_lat = model.a_lat
    b_lat = model.b_lat
    A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2
    V_uc = A_uc * thickness * (N_top + N_bottom)
    print(f"  Moiré unit cell area: {A_uc:.1f} Å²")
    print(f"  Unit cell volume: {V_uc:.1f} Å³")

    print(f"\n[3] Building BSE Hamiltonian...")
    H_bse = build_bse_hamiltonian(evals, evecs, k_points, v_idx, c_idx, A_uc,
                                   kappa=kappa, r0=r0, N_top=N_top, N_bottom=N_bottom)

    print(f"\n[4] Diagonalizing BSE ({H_bse.shape[0]}x{H_bse.shape[0]})...")
    import time
    t0 = time.time()
    dim_bse_mat = H_bse.shape[0]
    n_exciton_max = min(1000, dim_bse_mat - 2)
    if dim_bse_mat > 10000:
        print(f"    Using sparse eigsh (lowest {n_exciton_max} states)...")
        Omega_S, A_coeff = eigsh(H_bse, k=n_exciton_max, which='SM')
        sort_idx = np.argsort(Omega_S)
        Omega_S = Omega_S[sort_idx]
        A_coeff = A_coeff[:, sort_idx]
    else:
        Omega_S, A_coeff = scipy_eigh(H_bse, driver='evd')
    dt = time.time() - t0
    print(f"    Done in {dt:.1f} s")
    print(f"    Exciton energy range: {Omega_S[0]:.4f} - {Omega_S[-1]:.4f} eV")
    print(f"    Lowest exciton: {Omega_S[0]:.4f} eV  (QP gap ~ {np.min(dE):.4f} eV)")
    print(f"    Binding energy estimate: {np.min(dE) - Omega_S[0]:.4f} eV")

    # A_coeff[:, S] = expansion coefficients A^S_{vck}
    # BSE basis order: I = k * (Nv*Nc) + v * Nc + c
    dim_bse = Nv * Nc * Nk

    # 8. Compute exciton optical dipole and z-shift vector
    print(f"\n[5] Computing exciton observables...")

    # Flatten single-particle quantities to BSE basis order: [k, v, c]
    delta_z_flat = delta_z.reshape(dim_bse)

    omegas = np.linspace(E_range[0], E_range[1], n_E)
    results = {}

    for comp in comp_list:
        a_dir, b_dir, c_dir = comp
        assert b_dir == c_dir, "z-shift current only for linearly polarized light (b==c)"
        print(f"  sigma^{{{a_dir}{b_dir}{c_dir}}}...")

        r_b_flat = r_b[b_dir].reshape(dim_bse)  # (dim_bse,) complex

        # Gauge-invariant excitonic z-shift current formula:
        #   σ(ω) = (C/Nk) Σ_S Re[d^{b*}_S × g^{bz}_S] × δ(ω - Ω_S)
        #
        # where d^b_S  = Σ_{vck} A^S_{vck} r^b_{cv}(k)           (exciton dipole)
        #       g^{bz}_S = Σ_{vck} A^S_{vck} Δz(vck) r^b_{cv}(k)  (z-weighted dipole)
        #
        # This is gauge-invariant within degenerate exciton subspaces,
        # unlike the naive |d|² × R^z formula.

        d_b_S = A_coeff.conj().T @ r_b_flat  # (N_excitons,) complex
        g_bz_S = A_coeff.conj().T @ (delta_z_flat * r_b_flat)  # (N_excitons,) complex

        integrand_S = np.real(np.conj(d_b_S) * g_bz_S)  # (N_excitons,) real

        # F-sum rule check: BSE vs IPA integrated weights must match
        bse_sum = np.sum(integrand_S)
        ipa_sum = np.sum(np.real(delta_z_flat * np.abs(r_b_flat)**2))
        print(f"    F-sum check {a_dir}{b_dir}{c_dir}: BSE={bse_sum:.6e}, IPA={ipa_sum:.6e}, "
                f"ratio={bse_sum/ipa_sum:.6f}")

        # Assemble spectrum with Lorentzian broadening
        diff = omegas[:, None] - Omega_S[None, :]  # (n_E, N_excitons)
        lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
        sigma = lorentz @ integrand_S  # (n_E,)

        # 1/Nk BZ average
        sigma /= Nk
        results[comp] = sigma

    # 9. Physical prefactor (same as IPA)
    e_charge = 1.602176634e-19
    hbar = 1.054571817e-34
    prefactor = (2 * np.pi * e_charge**2) / (hbar * V_uc) * 1E6  # -> µA/V²

    for comp in comp_list:
        results[comp] *= prefactor

    # 10. Optionally compute IPA on the same grid for comparison
    ipa_results = None
    if plot_ipa_comparison:
        print(f"\n[6] Computing IPA comparison on same grid...")
        ipa_results = {}
        for comp in comp_list:
            a_dir, b_dir, c_dir = comp

            r_b_flat = r_b[b_dir].reshape(dim_bse)
            rb_sq = np.abs(r_b_flat)**2
            dE_flat = dE.reshape(dim_bse)

            integrand_ipa = delta_z.reshape(dim_bse) * rb_sq

            diff = omegas[:, None] - dE_flat[None, :]
            lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
            sigma_ipa = lorentz @ integrand_ipa / Nk

            sigma_ipa *= prefactor
            ipa_results[comp] = sigma_ipa

    # 11. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = {('z', 'x', 'x'): (r'$\sigma^{zxx}$', 'r'),
              ('z', 'y', 'y'): (r'$\sigma^{zyy}$', 'b')}

    for ax, comp in zip(axes, comp_list):
        lbl, col = labels[comp]
        ax.plot(omegas, results[comp], color=col, lw=2, label=f'BSE {lbl}')
        print(f"    {lbl}: BSE shift current max={np.max(abs(results[comp])):.2f} µA·Å/V²")
        if ipa_results is not None:
            ax.plot(omegas, ipa_results[comp], color=col, lw=1.5, ls='--', alpha=0.6,
                    label=f'IPA {lbl}')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel('Photon Energy (eV)')
        ax.set_ylabel(r'Shift Conductivity ($\mu$A/V$^2$)')
        ax.set_title(f'{lbl[:-1]}$ (BSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(E_range)

    fig.suptitle(f'Excitonic Z-Shift Current (BSE, Effective Model)\n'
                 f'N={N_top}/{N_bottom}, twist={np.degrees(twist_angle):.0f}°, '
                 f'kappa={kappa}, r0={r0} Å, d={thickness} Å, '
                 f'grid={n_k_bse}²', fontsize=11)
    plt.tight_layout()

    fname = f"EM_bse_z_sc{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved BSE Z-Shift Current Figure: {fname}")

    # 12. Exciton analysis plot
    osc_data = {}
    shift_weight_data = {}
    delta_z_flat = delta_z.reshape(dim_bse)
    for comp in comp_list:
        b_dir = comp[1]
        r_b_flat = r_b[b_dir].reshape(dim_bse)
        d_b_S = A_coeff.conj().T @ r_b_flat
        g_bz_S = A_coeff.conj().T @ (delta_z_flat * r_b_flat)
        osc_data[b_dir] = np.abs(d_b_S)**2
        shift_weight_data[b_dir] = np.real(np.conj(d_b_S) * g_bz_S)

    plot_exciton_analysis(Omega_S, osc_data, shift_weight_data, E_range=E_range,
                          N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle,
                          kappa=kappa, r0=r0, thickness=thickness,
                          save_prefix=save_prefix)

    return omegas, results, Omega_S, A_coeff


# =====================================================================
#  Full Excitonic Shift Current — Eq. (10), Huang/Chan/Guo PRB 108 075413
# =====================================================================

def _enforce_c2_gauge(evecs, n_k_side):
    r"""
    Enforce C2-symmetric gauge: evecs(-k) = D @ evecs(k).

    D = diag(sigma_x, sigma_x) satisfies D H(-k) D^\dagger = H(k).
    """
    D = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    evecs_c2 = evecs.copy()
    N = n_k_side
    for i in range(N):
        for j in range(N):
            ip, jp = N - 1 - i, N - 1 - j
            idx = i * N + j
            idx_p = ip * N + jp
            if idx_p > idx:
                evecs_c2[idx_p] = D @ evecs_c2[idx]
    return evecs_c2


def _enforce_c4_gauge(evecs, n_k_side):
    r"""
    Enforce C4-symmetric gauge for 90-degree twisted bilayer BP.

    The C4 operator S = [[0, I_2], [sigma_x, 0]] satisfies
    S H(kx,ky) S^\dagger = H(-ky, kx), with S^2 = C2 = D.

    For each C4 orbit {k, C4 k, C4^2 k, C4^3 k}, eigenvectors at the
    representative k-point are kept from eigh, and the rest are set via
    evecs(C4^n k) = S^n @ evecs(k).

    This enforces both C2 (forbidden in-plane components = 0) and C4
    (sigma^{zxx} = sigma^{zyy}).

    Grid mapping under C4: (i, j) -> (N-1-j, i).

    Parameters
    ----------
    evecs : ndarray, shape (Nk, dim_H, dim_H)
        Eigenvectors on a square k-grid of size n_k_side x n_k_side.
    n_k_side : int
        Grid size per direction (total Nk = n_k_side^2).

    Returns
    -------
    evecs_c4 : ndarray, same shape as evecs
        Eigenvectors with C4-symmetric gauge.
    """
    # C4 operator: S = [[0, I_2], [sigma_x, 0]], S^2 = D (C2)
    S = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0]], dtype=complex)
    S2 = S @ S   # = D (C2)
    S3 = S2 @ S  # = S^3

    evecs_c4 = evecs.copy()
    N = n_k_side
    visited = set()

    for i in range(N):
        for j in range(N):
            idx0 = i * N + j
            if idx0 in visited:
                continue

            # C4 orbit: (i,j) -> (N-1-j, i) -> (N-1-i, N-1-j) -> (j, N-1-i)
            orbit_ij = [
                (i, j),
                (N - 1 - j, i),
                (N - 1 - i, N - 1 - j),
                (j, N - 1 - i),
            ]
            orbit_idx = [ii * N + jj for ii, jj in orbit_ij]

            # Mark all as visited
            for idx in orbit_idx:
                visited.add(idx)

            # Keep representative (first in orbit), set others via S^n
            rep = orbit_idx[0]
            S_powers = [np.eye(4, dtype=complex), S, S2, S3]
            for n in range(1, 4):
                if orbit_idx[n] != rep:
                    evecs_c4[orbit_idx[n]] = S_powers[n] @ evecs_c4[rep]

    return evecs_c4


def _compute_z_velocity_in_eigenbasis(H_stack, evecs, z_op):
    r"""
    Compute z-direction velocity operator v^z = [H, z]/(i\hbar) in eigenbasis.

    In a 2D system, v^z is NOT dH/dk_z. Instead it is the commutator:
        v^z_{\mu\nu} = -i * H_{\mu\nu} * (z_\nu - z_\mu)
    in the orbital basis (with \hbar = 1, giving units eV*Angstrom).

    Parameters
    ----------
    H_stack : ndarray, shape (Nk, Nb, Nb)
        Hamiltonian matrices in orbital basis.
    evecs : ndarray, shape (Nk, Nb, Nb)
        Eigenvectors from eigh (columns are eigenstates).
    z_op : ndarray, shape (Nb, Nb)
        Diagonal z-position operator in orbital basis (Angstrom).

    Returns
    -------
    vz_eig : ndarray, shape (Nk, Nb, Nb), complex
        z-velocity matrix in eigenbasis.
    """
    z_diag_orb = np.diag(z_op)  # (Nb,)
    # z_diff[mu, nu] = z_nu - z_mu
    z_diff = z_diag_orb[np.newaxis, :] - z_diag_orb[:, np.newaxis]  # (Nb, Nb)
    # v^z_{mu,nu}(k) = -i * H_{mu,nu}(k) * (z_nu - z_mu)
    vz_orb = -1j * H_stack * z_diff[np.newaxis, :, :]  # (Nk, Nb, Nb)
    U_dag = np.conj(np.transpose(evecs, (0, 2, 1)))
    vz_eig = U_dag @ vz_orb @ evecs
    return vz_eig


def _compute_exciton_velocity(A_coeff, v_a_eig, v_idx, c_idx, n_exc):
    r"""
    Compute exciton velocity coupling V^a_s = \sum_{cvk} A^{s*}_{cvk} v^a_{cv}(k).

    This uses the velocity matrix element directly (NOT the position r = v/(iw)).

    Parameters
    ----------
    A_coeff : ndarray, shape (dim_bse, N_excitons)
        BSE eigenvectors.
    v_a_eig : ndarray, shape (Nk, Nb, Nb)
        Velocity matrix in eigenbasis for direction a.
    v_idx, c_idx : array of int
    n_exc : int

    Returns
    -------
    V_a_s : ndarray, shape (n_exc,), complex
    """
    Nv, Nc = len(v_idx), len(c_idx)
    # Extract v^a_{cv}(k): indexed as v_a_eig[:, c, v]
    va_cv = v_a_eig[:, c_idx, :][:, :, v_idx]  # (Nk, Nc, Nv)
    va_cv = np.transpose(va_cv, (0, 2, 1))      # (Nk, Nv, Nc) — BSE order
    va_cv_flat = va_cv.reshape(-1)               # (dim_bse,)
    V_a_s = A_coeff[:, :n_exc].conj().T @ va_cv_flat  # (n_exc,)
    return V_a_s


def _compute_Delta_matrix(A_coeff, v_a_eig, v_idx, c_idx, Nk, n_exc):
    r"""
    Compute inter-exciton velocity matrix (intraband velocity):

    \Delta^a_{ss'} = \sum_{cc'vk} A^s_{cvk} v^a_{c'c}(k) A^{s'*}_{c'vk}
                   - \sum_{cvv'k} A^s_{cv'k} v^a_{v'v}(k) A^{s'*}_{cvk}

    Parameters
    ----------
    A_coeff : ndarray, shape (dim_bse, N_excitons)
    v_a_eig : ndarray, shape (Nk, Nb, Nb)
        Full velocity matrix in eigenbasis for direction a.
    v_idx, c_idx : array of int
    Nk : int
    n_exc : int

    Returns
    -------
    Delta_a : ndarray, shape (n_exc, n_exc), complex (Hermitian)
    """
    Nv, Nc = len(v_idx), len(c_idx)
    A_trunc = A_coeff[:, :n_exc]
    Delta_a = np.zeros((n_exc, n_exc), dtype=np.complex128)

    for ik in range(Nk):
        blk = slice(ik * Nv * Nc, (ik + 1) * Nv * Nc)
        A_k = A_trunc[blk, :].reshape(Nv, Nc, n_exc)

        # Electron (conduction-conduction) term: + sum_v (phi^v)^H @ va_cc @ phi^v
        va_cc = v_a_eig[ik][np.ix_(c_idx, c_idx)]  # (Nc, Nc)
        for iv in range(Nv):
            phi = A_k[iv, :, :]  # (Nc, n_exc)
            Delta_a += phi.conj().T @ va_cc @ phi

        # Hole (valence-valence) term: - sum_c (psi^c)^H @ va_vv @ psi^c
        va_vv = v_a_eig[ik][np.ix_(v_idx, v_idx)]  # (Nv, Nv)
        for ic in range(Nc):
            psi = A_k[:, ic, :]  # (Nv, n_exc)
            Delta_a -= psi.conj().T @ va_vv @ psi

    return Delta_a


def _compute_Y_matrix(A_coeff, v_b_eig, evals, v_idx, c_idx, Nk, n_exc):
    r"""
    Compute interband part of inter-exciton position matrix:

    Y^b_{ss'} = \sum_{c'\neq c, v, k} A^{s*}_{c'vk} x^b_{c'c}(k) A^{s'}_{cvk}
              - \sum_{c, v'\neq v, k} A^{s*}_{cv'k} x^b_{v'v}(k) A^{s'}_{cvk}

    where x^b_{mn} = v^b_{mn} / (i * omega_{mn}) for m != n (interband position).

    Parameters
    ----------
    A_coeff : ndarray, shape (dim_bse, N_excitons)
    v_b_eig : ndarray, shape (Nk, Nb, Nb)
        Velocity matrix in eigenbasis for direction b.
    evals : ndarray, shape (Nk, Nb)
        Eigenvalues.
    v_idx, c_idx : array of int
    Nk : int
    n_exc : int

    Returns
    -------
    Y_b : ndarray, shape (n_exc, n_exc), complex
    """
    Nv, Nc = len(v_idx), len(c_idx)
    A_trunc = A_coeff[:, :n_exc]
    Y_b = np.zeros((n_exc, n_exc), dtype=np.complex128)
    eps_denom = 1e-5

    for ik in range(Nk):
        blk = slice(ik * Nv * Nc, (ik + 1) * Nv * Nc)
        A_k = A_trunc[blk, :].reshape(Nv, Nc, n_exc)

        # Conduction-conduction interband position x^b_{c'c}
        E_c = evals[ik, c_idx]  # (Nc,)
        dE_cc = E_c[:, np.newaxis] - E_c[np.newaxis, :]  # (Nc, Nc)
        vb_cc = v_b_eig[ik][np.ix_(c_idx, c_idx)]        # (Nc, Nc)
        xb_cc = np.zeros_like(vb_cc)
        off_diag = np.abs(dE_cc) > eps_denom
        xb_cc[off_diag] = vb_cc[off_diag] / (1j * dE_cc[off_diag])
        np.fill_diagonal(xb_cc, 0.0)

        for iv in range(Nv):
            phi = A_k[iv, :, :]  # (Nc, n_exc)
            Y_b += phi.conj().T @ xb_cc @ phi

        # Valence-valence interband position x^b_{v'v}
        E_v = evals[ik, v_idx]  # (Nv,)
        dE_vv = E_v[:, np.newaxis] - E_v[np.newaxis, :]  # (Nv, Nv)
        vb_vv = v_b_eig[ik][np.ix_(v_idx, v_idx)]        # (Nv, Nv)
        xb_vv = np.zeros_like(vb_vv)
        off_diag_v = np.abs(dE_vv) > eps_denom
        xb_vv[off_diag_v] = vb_vv[off_diag_v] / (1j * dE_vv[off_diag_v])
        np.fill_diagonal(xb_vv, 0.0)

        for ic in range(Nc):
            psi = A_k[:, ic, :]  # (Nv, n_exc)
            Y_b -= psi.conj().T @ xb_vv @ psi

    return Y_b


def _compute_Q_matrix(A_coeff, evecs, v_idx, c_idx, n_k_bse, n_exc, k_range):
    r"""
    Compute intraband (Berry connection) part of inter-exciton position:

    Q^b_{ss'} = i \sum_{cvk} A^{s*}_{cvk} [D_b A^{s'}]_{cvk}

    where D_b is the covariant k-derivative using parallel transport
    via overlap matrices to ensure gauge invariance.

    Parameters
    ----------
    A_coeff : ndarray, shape (dim_bse, N_excitons)
    evecs : ndarray, shape (Nk, Nb, Nb)
    v_idx, c_idx : array of int
    n_k_bse : int
        k-grid size per direction (total Nk = n_k_bse^2).
    n_exc : int
    k_range : float
        Half-extent of k-grid in each direction.

    Returns
    -------
    Q : dict with keys 'x', 'y', each ndarray shape (n_exc, n_exc), complex
    """
    Nv, Nc = len(v_idx), len(c_idx)
    Nk = n_k_bse * n_k_bse
    dim_vc = Nv * Nc
    A_trunc = A_coeff[:, :n_exc]  # (dim_bse, n_exc)

    dk = 2.0 * k_range / (n_k_bse - 1) if n_k_bse > 1 else 1.0

    # Extract eigenvectors for active bands
    U_c = evecs[:, :, c_idx]  # (Nk, Nb, Nc)
    U_v = evecs[:, :, v_idx]  # (Nk, Nb, Nv)

    Q = {}
    # directions: 'x' shifts along ix (stride 1), 'y' shifts along iy (stride n_k_bse)
    for b_dir, stride in [('x', 1), ('y', n_k_bse)]:
        Q_b = np.zeros((n_exc, n_exc), dtype=np.complex128)

        # Accumulate covariant derivative contribution for each k-point
        for ik in range(Nk):
            iy, ix = divmod(ik, n_k_bse)

            # Determine neighbors and finite difference type
            if b_dir == 'x':
                at_lo, at_hi = (ix == 0), (ix == n_k_bse - 1)
            else:
                at_lo, at_hi = (iy == 0), (iy == n_k_bse - 1)

            # Get A at current k
            blk_k = slice(ik * dim_vc, (ik + 1) * dim_vc)
            A_k = A_trunc[blk_k, :]  # (dim_vc, n_exc)

            D_b_A = np.zeros_like(A_k)

            if not at_hi:
                # Forward neighbor
                ik_plus = ik + stride
                blk_plus = slice(ik_plus * dim_vc, (ik_plus + 1) * dim_vc)
                A_plus = A_trunc[blk_plus, :]  # (dim_vc, n_exc)

                # Overlap matrices: transport from k+db to k
                O_c_plus = U_c[ik].conj().T @ U_c[ik_plus]  # (Nc, Nc)
                O_v_plus = U_v[ik].conj().T @ U_v[ik_plus]  # (Nv, Nv)

                # Parallel transport: T = conj(O_v) kron O_c
                A_plus_3d = A_plus.reshape(Nv, Nc, n_exc)
                # transported[v, c, s] = sum_{c'v'} O_c[c,c'] conj(O_v[v,v']) A[v',c',s]
                transported_plus = np.einsum('ij,kl,ljn->kin',
                                             O_c_plus, O_v_plus.conj(),
                                             A_plus_3d)
                transported_plus = transported_plus.reshape(dim_vc, n_exc)

            if not at_lo:
                # Backward neighbor
                ik_minus = ik - stride
                blk_minus = slice(ik_minus * dim_vc, (ik_minus + 1) * dim_vc)
                A_minus = A_trunc[blk_minus, :]

                O_c_minus = U_c[ik].conj().T @ U_c[ik_minus]  # (Nc, Nc)
                O_v_minus = U_v[ik].conj().T @ U_v[ik_minus]  # (Nv, Nv)

                A_minus_3d = A_minus.reshape(Nv, Nc, n_exc)
                transported_minus = np.einsum('ij,kl,ljn->kin',
                                              O_c_minus, O_v_minus.conj(),
                                              A_minus_3d)
                transported_minus = transported_minus.reshape(dim_vc, n_exc)

            # Finite difference
            if at_lo and not at_hi:
                D_b_A = (transported_plus - A_k) / dk
            elif at_hi and not at_lo:
                D_b_A = (A_k - transported_minus) / dk
            elif not at_lo and not at_hi:
                D_b_A = (transported_plus - transported_minus) / (2.0 * dk)
            # else: single point, D_b_A stays zero

            # Accumulate: Q^b[s, s'] += i * A_k^H @ D_b_A
            Q_b += 1j * (A_k.conj().T @ D_b_A)

        Q[b_dir] = Q_b

    return Q


def _assemble_excitonic_spectrum(X_b_s, X_c_s, V_a_s, Delta_a,
                                  X_b_ss, X_c_ss, Omega_S,
                                  omegas, eta):
    r"""
    Assemble excitonic shift current spectrum from Eq. (10).

    sigma^{abc}(omega) = sum_{ss'} {
        Term1: X^b_s Delta^a_{ss'} X^{c*}_{s'} / [(w-Os+ie)(w-Os'-ie)]
        Term2: V^{a*}_s X^b_{ss'} X^c_{s'} / [(-Os)(w-Os'+ie)]
        Term3: V^a_s X^{b*}_{ss'} X^{c*}_{s'} / [(-Os)(-w-Os'-ie)]
    } + (b<->c, w<->-w)

    Parameters
    ----------
    X_b_s, X_c_s : ndarray (n_exc,), complex — exciton optical dipoles
    V_a_s : ndarray (n_exc,), complex — exciton velocity
    Delta_a : ndarray (n_exc, n_exc), complex — inter-exciton velocity matrix
    X_b_ss, X_c_ss : ndarray (n_exc, n_exc), complex — inter-exciton position
    Omega_S : ndarray (n_exc,), real — exciton energies
    omegas : ndarray (n_E,), real — photon energies
    eta : float — broadening

    Returns
    -------
    sigma : ndarray (n_E,), real — shift current spectrum (before prefactor)
    """
    n_exc = len(Omega_S)
    n_E = len(omegas)

    # Safety: avoid division by zero for Omega_s
    safe_Omega = np.where(np.abs(Omega_S) > 1e-8, Omega_S, 1e-8)

    def _compute_three_terms(Xb, Xc, Va, Xb_ss_mat, Xc_ss_mat, w_sign=1.0):
        """Compute the three terms for given polarizations and frequency sign."""
        w = w_sign * omegas  # (n_E,)

        # Frequency denominators
        # L_s = 1/(w - Omega_s + i*eta), shape (n_E, n_exc)
        L = 1.0 / (w[:, None] - Omega_S[None, :] + 1j * eta)
        # R_s' = 1/(w - Omega_{s'} - i*eta)
        R = 1.0 / (w[:, None] - Omega_S[None, :] - 1j * eta)

        # --- Term 1: X^b_s Delta^a_{ss'} X^{c*}_{s'} L_s R_{s'} ---
        # L_weighted[e, s] = X^b_s * L_s(omega_e)
        L_weighted = Xb[None, :] * L   # (n_E, n_exc)
        R_weighted = np.conj(Xc)[None, :] * R  # (n_E, n_exc)
        # term1 = sum_{ss'} L_weighted @ Delta @ R_weighted
        term1 = np.sum((L_weighted @ Delta_a) * R_weighted, axis=1)  # (n_E,)

        # --- Term 2: V^{a*}_s X^b_{ss'} X^c_{s'} / [(-Omega_s)(w - Omega_{s'} + ie)] ---
        F2 = np.conj(Va) / (-safe_Omega)  # (n_exc,)
        g2 = F2 @ Xb_ss_mat                # (n_exc,)  [sum over s]
        L2 = 1.0 / (w[:, None] - Omega_S[None, :] + 1j * eta)  # (n_E, n_exc)
        term2 = L2 @ (g2 * Xc)  # (n_E,)

        # --- Term 3: V^a_s X^{b*}_{ss'} X^{c*}_{s'} / [(-Omega_s)(-w - Omega_{s'} - ie)] ---
        F3 = Va / (-safe_Omega)             # (n_exc,)
        g3 = F3 @ np.conj(Xb_ss_mat)       # (n_exc,)  [sum over s]
        R3 = 1.0 / (-w[:, None] - Omega_S[None, :] - 1j * eta)  # (n_E, n_exc)
        term3 = R3 @ (g3 * np.conj(Xc))  # (n_E,)

        return term1 + term2 + term3

    # Original terms: (b, c, omega)
    sigma_orig = _compute_three_terms(X_b_s, X_c_s, V_a_s,
                                       X_b_ss, X_c_ss, w_sign=+1.0)

    # Symmetrization: (c, b, -omega) — swap b<->c and omega -> -omega
    sigma_swap = _compute_three_terms(X_c_s, X_b_s, V_a_s,
                                       X_c_ss, X_b_ss, w_sign=-1.0)

    # For b = c (linearly polarized light), the sum is real.
    # For the general case, the shift current is the real part of
    # the symmetrized expression.
    sigma = np.real(sigma_orig + sigma_swap)

    return sigma


def calculate_excitonic_shift_current(
        N_top=1, N_bottom=1, twist_angle=0.0,
        E_range=(0.0, 1.0), n_E=400, eta=0.010,
        k_range=0.15, n_k_bse=30,
        n_val=2, n_cond=2,
        thickness=5.2,
        kappa=2.5, r0=5.0,
        n_exciton_keep=200,
        components=None,
        plot_ipa_comparison=True,
        band_window=None, save_prefix=""):
    r"""
    Full excitonic shift current via Eq. (10) of Huang, Chan, and Guo,
    PRB 108, 075413 (2023).

    Computes both z-shift current (\sigma^{zbb}) and in-plane shift current
    (\sigma^{abb}) using exciton eigenstates from BSE with the full
    inter-exciton coupling formalism.

    Parameters
    ----------
    N_top, N_bottom : int
        Number of top/bottom layers.
    twist_angle : float
        Twist angle in radians.
    E_range : tuple
        (E_min, E_max) for photon energy axis.
    n_E : int
        Number of energy points.
    eta : float
        Lorentzian broadening in eV.
    k_range : float
        Half-extent of k-grid (Angstrom^-1).
    n_k_bse : int
        k-grid size per direction (total Nk = n_k_bse^2).
    n_val, n_cond : int
        Number of valence/conduction bands in BSE active space.
    thickness : float
        Layer thickness in Angstrom.
    kappa : float
        Dielectric constant for Keldysh potential.
    r0 : float
        2D polarizability screening length in Angstrom.
    n_exciton_keep : int
        Number of lowest excitons to retain.
    components : list of tuples, optional
        List of (a, b, c) components. Default includes z and in-plane.
    plot_ipa_comparison : bool
        Overlay IPA shift current for comparison.
    """
    if components is None:
        # Only z-shift components are symmetry-allowed for C2-symmetric systems.
        # All purely in-plane components (xxx, yyy, xyy, yxx) vanish by
        # C2 inversion: sigma^{abc} -> (-1)^3 sigma^{abc} = 0.
        components = [
            ('z', 'x', 'x'), ('z', 'y', 'y'), ('z', 'x', 'y'),
            ('x', 'y', 'y'), ('x', 'x', 'x'), ('y', 'x', 'x')
        ]

    print("=" * 60)
    print("Full Excitonic Shift Current — Eq. (10) PRB 108 075413")
    print("=" * 60)
    print(f"  Grid: {n_k_bse}x{n_k_bse} = {n_k_bse**2} k-points")
    print(f"  Active space: {n_val}v x {n_cond}c")
    print(f"  BSE dimension: {n_val * n_cond * n_k_bse**2}")
    print(f"  Exciton truncation: {n_exciton_keep}")
    print(f"  Keldysh params: kappa={kappa}, r0={r0} Å")
    print(f"  Components: {components}")

    # ---- 1. Model & k-grid ----
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)
    kx = np.linspace(-k_range, k_range, n_k_bse)
    ky = np.linspace(-k_range, k_range, n_k_bse)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    # ---- 2. Single-particle diagonalization ----
    print(f"\n[1] Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)
    Nb = evals.shape[1]

    mid = Nb // 2
    if band_window is not None:
        v_idx = np.arange(band_window[0], band_window[1] + 1)
        c_idx = np.arange(band_window[2], band_window[3] + 1)
    else:
        v_idx = np.arange(mid - n_val, mid)
        c_idx = np.arange(mid, mid + n_cond)
    Nv, Nc = len(v_idx), len(c_idx)
    print(f"  Bands: valence {v_idx}, conduction {c_idx}")

    # ---- 2b. Enforce C2-symmetric gauge ----
    # C2 gauge: evecs(-k) = D @ evecs(k), ensures forbidden components = 0.
    # C4 gauge was tried but makes Q (parallel transport) less accurate
    # because S mixes sublattices, creating a less smooth gauge.
    # The residual sigma^{zxx} != sigma^{zyy} is a finite-grid artifact.
    evecs = _enforce_c2_gauge(evecs, n_k_bse)

    # ---- 3. Velocity and position matrices ----
    print(f"\n[2] Computing velocity, position, and z-velocity matrices...")
    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))
    vx_eig = U_dag @ vx_orb @ U
    vy_eig = U_dag @ vy_orb @ U

    # z-position operator
    dim_H = Nb
    z_op = np.zeros((dim_H, dim_H), dtype=np.float64)
    z_op[0, 0] = +thickness * N_top / 2.0
    z_op[1, 1] = +thickness * N_top / 2.0
    if dim_H == 4:
        z_op[2, 2] = -thickness * N_bottom / 2.0
        z_op[3, 3] = -thickness * N_bottom / 2.0

    # z-velocity
    vz_eig = _compute_z_velocity_in_eigenbasis(H_stack, evecs, z_op)

    # Velocity map for all directions
    v_eig_map = {'x': vx_eig, 'y': vy_eig, 'z': vz_eig}

    # Interband position r^b_{cv}(k) = v^b_{cv}/(i * omega_{cv})
    E_v_arr = evals[:, v_idx]  # (Nk, Nv)
    E_c_arr = evals[:, c_idx]  # (Nk, Nc)
    dE = E_c_arr[:, None, :] - E_v_arr[:, :, None]  # (Nk, Nv, Nc)
    eps_denom = 1e-5

    r_b = {}
    for b_dir in ['x', 'y']:
        v_b = v_eig_map[b_dir]
        vb_cv = v_b[:, c_idx, :][:, :, v_idx]   # (Nk, Nc, Nv)
        vb_cv = np.transpose(vb_cv, (0, 2, 1))  # (Nk, Nv, Nc)
        rb = np.zeros_like(vb_cv)
        valid = np.abs(dE) > eps_denom
        rb[valid] = vb_cv[valid] / (1j * dE[valid])
        r_b[b_dir] = rb

    # ---- 4. BSE ----
    a_lat = model.a_lat
    b_lat = model.b_lat
    A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2
    V_uc = A_uc * thickness * (N_top + N_bottom)
    print(f"  Unit cell area: {A_uc:.1f} Å², volume: {V_uc:.1f} Å³")

    print(f"\n[3] Building BSE Hamiltonian...")
    H_bse = build_bse_hamiltonian(evals, evecs, k_points, v_idx, c_idx, A_uc,
                                   kappa=kappa, r0=r0,
                                   N_top=N_top, N_bottom=N_bottom)

    print(f"\n[4] Diagonalizing BSE ({H_bse.shape[0]}x{H_bse.shape[0]})...")
    import time
    t0 = time.time()
    dim_bse = H_bse.shape[0]
    if dim_bse > 10000:
        n_eig = min(n_exciton_keep + 50, dim_bse - 2)
        print(f"    Using sparse eigsh (lowest {n_eig} states)...")
        Omega_S, A_coeff = eigsh(H_bse, k=n_eig, which='SM')
        sort_idx = np.argsort(Omega_S)
        Omega_S = Omega_S[sort_idx]
        A_coeff = A_coeff[:, sort_idx]
    else:
        Omega_S, A_coeff = scipy_eigh(H_bse, driver='evd')
    dt = time.time() - t0
    print(f"    Done in {dt:.1f} s")

    # Truncate to n_exciton_keep
    n_exc = min(n_exciton_keep, len(Omega_S))
    Omega_trunc = Omega_S[:n_exc]
    A_trunc = A_coeff[:, :n_exc]
    print(f"    Keeping {n_exc} lowest excitons: {Omega_trunc[0]:.4f} - {Omega_trunc[-1]:.4f} eV")
    print(f"    Binding energy estimate: {np.min(dE) - Omega_trunc[0]:.4f} eV")

    # ---- 5. Build all inter-exciton matrices ----
    print(f"\n[5] Computing inter-exciton matrices...")

    # X^b_s (exciton optical dipole) for b = x, y
    # Paper Eq. (5): X^b_s = Σ_{cvk} A^s_{cvk} x^b_{vc}(k)
    # where x_{vc} = conj(x_{cv}) = conj(r_b).
    # So X^b_s = A^T @ conj(r_b_flat), NOT A^† @ r_b_flat.
    X_b_s = {}
    for b_dir in ['x', 'y']:
        r_b_flat = r_b[b_dir].reshape(-1)
        X_b_s[b_dir] = A_trunc.T @ np.conj(r_b_flat)  # (n_exc,)

    # V^a_s (exciton velocity) for a = x, y, z
    V_a_s = {}
    for a_dir in ['x', 'y', 'z']:
        V_a_s[a_dir] = _compute_exciton_velocity(A_trunc, v_eig_map[a_dir],
                                                   v_idx, c_idx, n_exc)
        print(f"    V^{a_dir}_s: max |V| = {np.max(np.abs(V_a_s[a_dir])):.4e}")

    # Delta^a_{ss'} (inter-exciton velocity) for a = x, y, z
    Delta_a = {}
    for a_dir in ['x', 'y', 'z']:
        Delta_a[a_dir] = _compute_Delta_matrix(A_trunc, v_eig_map[a_dir],
                                                v_idx, c_idx, Nk, n_exc)
        herm_err = np.max(np.abs(Delta_a[a_dir] - Delta_a[a_dir].conj().T))
        print(f"    Delta^{a_dir}: Hermiticity error = {herm_err:.2e}")

    # Y^b_{ss'} (interband inter-exciton position) for b = x, y
    Y_b = {}
    for b_dir in ['x', 'y']:
        Y_b[b_dir] = _compute_Y_matrix(A_trunc, v_eig_map[b_dir], evals,
                                         v_idx, c_idx, Nk, n_exc)
        print(f"    Y^{b_dir}: max |Y| = {np.max(np.abs(Y_b[b_dir])):.4e}")

    # Q^b_{ss'} (intraband inter-exciton position) for b = x, y
    print(f"    Computing Q (gauge-invariant k-derivatives)...")
    Q_b = _compute_Q_matrix(A_trunc, evecs, v_idx, c_idx,
                             n_k_bse, n_exc, k_range)
    for b_dir in ['x', 'y']:
        print(f"    Q^{b_dir}: max |Q| = {np.max(np.abs(Q_b[b_dir])):.4e}")

    # X^b_{ss'} = Y^b + Q^b
    X_b_ss = {}
    for b_dir in ['x', 'y']:
        X_b_ss[b_dir] = Y_b[b_dir] + Q_b[b_dir]
        print(f"    X^{b_dir}_{'{ss}'}: max |X| = {np.max(np.abs(X_b_ss[b_dir])):.4e}")

    # ---- 6. Assemble spectra ----
    print(f"\n[6] Assembling excitonic shift current spectra...")
    omegas = np.linspace(E_range[0], E_range[1], n_E)
    results = {}

    # Physical prefactor: -(gs * e^3) / (2 * Nk * Vc)
    # Convert to code units -> microamps*Angstrom/V^2
    gs = 2  # spin degeneracy
    e_charge = 1.602176634e-19   # C
    hbar = 1.054571817e-34       # J*s
    # sigma has dimensions of [position * charge / (electric field)^2]
    # In atomic-like units (eV, Angstrom), the formula gives sigma in eV*Angstrom^2
    # To get muA*Angstrom/V^2:
    # sigma_SI = (gs * e^3) / (2 * hbar * Nk * V_uc) * sigma_code
    # The factor of 1/hbar converts from eV^-1 to s (since omega is in eV/hbar)
    prefactor = -gs * e_charge**2 / (2.0 * hbar * Nk * V_uc) * 1e6  # muA*Angstrom/V^2

    for comp in components:
        a_dir, b_dir, c_dir = comp
        print(f"  sigma^{{{a_dir}{b_dir}{c_dir}}}...")

        sigma = _assemble_excitonic_spectrum(
            X_b_s=X_b_s[b_dir], X_c_s=X_b_s[c_dir],
            V_a_s=V_a_s[a_dir], Delta_a=Delta_a[a_dir],
            X_b_ss=X_b_ss[b_dir], X_c_ss=X_b_ss[c_dir],
            Omega_S=Omega_trunc, omegas=omegas, eta=eta)

        sigma *= prefactor
        results[comp] = sigma
        print(f"    max |sigma| = {np.max(np.abs(sigma)):.4f} µA·Å/V²")

    # ---- 7. IPA comparison ----
    ipa_results = None
    if plot_ipa_comparison:
        print(f"\n[7] Computing IPA comparison...")
        ipa_results = {}

        # z-operator diagonal in eigenbasis
        z_eig = U_dag @ z_op @ U
        z_diag = np.real(np.diagonal(z_eig, axis1=1, axis2=2))
        z_v = z_diag[:, v_idx]
        z_c = z_diag[:, c_idx]
        delta_z = z_v[:, :, None] - z_c[:, None, :]  # (Nk, Nv, Nc)

        # Also need w_xx, w_yy, w_xy for in-plane IPA shift current
        w_xx_orb, w_yy_orb, w_xy_orb = model.get_generalized_derivative_matrices(k_points)
        w_map = {
            'xx': U_dag @ w_xx_orb @ U,
            'yy': U_dag @ w_yy_orb @ U,
            'xy': U_dag @ w_xy_orb @ U,
            'yx': U_dag @ w_xy_orb @ U,
        }

        for comp in components:
            a_dir, b_dir, c_dir = comp

            if a_dir == 'z':
                # IPA z-shift: sigma ∝ sum_{nm} |r^b_{nm}|^2 * delta_z * delta(w - w_nm)
                r_b_cv = r_b[b_dir]  # (Nk, Nv, Nc)
                rb_sq = np.abs(r_b_cv)**2
                integrand = (-1) * delta_z * rb_sq
                integrand_flat = integrand.reshape(Nk, -1)
                dE_flat = dE.reshape(Nk, -1)
                N_pairs = integrand_flat.shape[1]

                sigma_ipa = np.zeros_like(omegas)
                for b_start in range(0, N_pairs, max(1, 200_000_000 // (n_E * Nk))):
                    b_end = min(b_start + max(1, 200_000_000 // (n_E * Nk)), N_pairs)
                    diff = omegas[:, None, None] - dE_flat[None, :, b_start:b_end]
                    lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
                    sigma_ipa += np.sum(lorentz * integrand_flat[None, :, b_start:b_end], axis=(1, 2))
                sigma_ipa /= Nk
                # Prefactor for IPA z-shift
                prefactor_ipa_z = (2 * np.pi * e_charge**2) / (hbar * V_uc) * 1e6
                sigma_ipa *= prefactor_ipa_z

            else:
                # IPA in-plane shift current via SOS method
                v_a = v_eig_map[a_dir]
                v_b = v_eig_map[b_dir]
                v_c = v_eig_map[c_dir]
                w_ac = w_map[a_dir + c_dir]
                w_all = evals[:, :, None] - evals[:, None, :]

                sigma_ipa = np.zeros_like(omegas)
                for n in c_idx:
                    for m in v_idx:
                        f_nm = -1.0  # f_cond - f_val = 0 - 1 = -1
                        w_nm = evals[:, n] - evals[:, m]
                        nonzero = w_nm > eps_denom

                        r_b_mn = np.zeros(Nk, dtype=np.complex128)
                        r_b_mn[nonzero] = v_b[nonzero, m, n] / (-1j * w_nm[nonzero])

                        # Term A
                        termA = np.zeros(Nk, dtype=np.complex128)
                        delta_a_nm = v_a[nonzero, n, n] - v_a[nonzero, m, m]
                        delta_c_nm = v_c[nonzero, n, n] - v_c[nonzero, m, m]
                        termA[nonzero] = (v_c[nonzero, n, m] * delta_a_nm
                                        + v_a[nonzero, n, m] * delta_c_nm) / w_nm[nonzero]

                        # Term B
                        w_np = w_all[:, n, :]
                        w_pm = w_all[:, :, m]
                        valid_p = (np.abs(w_np) > eps_denom) & (np.abs(w_pm) > eps_denom)
                        valid_p[:, n] = False; valid_p[:, m] = False
                        valid_p &= nonzero[:, None]
                        num1 = v_c[:, n, :] * v_a[:, :, m]
                        num2 = v_a[:, n, :] * v_c[:, :, m]
                        termB_c = np.zeros((Nk, Nb), dtype=np.complex128)
                        termB_c[valid_p] = (num1[valid_p] / w_pm[valid_p]
                                          - num2[valid_p] / w_np[valid_p])
                        termB = np.sum(termB_c, axis=1)

                        # Term C
                        termC = -w_ac[:, n, m]

                        K_nm = termA + termB + termC
                        r_deriv = np.zeros(Nk, dtype=np.complex128)
                        r_deriv[nonzero] = K_nm[nonzero] / (-1j * w_nm[nonzero])

                        weight = f_nm * np.imag(r_b_mn * r_deriv)
                        diff = omegas[:, None] - w_nm[None, :]
                        lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
                        sigma_ipa += np.sum(lorentz * weight[None, :], axis=1)

                sigma_ipa /= Nk
                prefactor_ipa = (2 * np.pi * e_charge**2) / (hbar * A_uc) * 1e6
                sigma_ipa *= prefactor_ipa

            ipa_results[comp] = sigma_ipa

    # ---- 8. Plotting ----
    n_plots = len(components)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    colors = {'x': 'r', 'y': 'b', 'z': 'g'}
    for idx, comp in enumerate(components):
        ax = axes[idx // n_cols, idx % n_cols]
        a_dir, b_dir, c_dir = comp
        label = rf'$\sigma^{{{a_dir}{b_dir}{c_dir}}}$'
        col = colors.get(a_dir, 'k')

        ax.plot(omegas, results[comp], color=col, lw=2, label=f'BSE {label}')
        if ipa_results is not None and comp in ipa_results:
            ax.plot(omegas, ipa_results[comp], color=col, lw=1.5, ls='--',
                    alpha=0.6, label=f'IPA {label}')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel('Photon Energy (eV)')
        ax.set_ylabel(r'Shift Conductivity ($\mu$A$\cdot$Å/V$^2$)')
        ax.set_title(f'{label} (BSE Eq.10)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(E_range)

    # Hide unused axes
    for idx in range(n_plots, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(f'Excitonic Shift Current (Full Eq.10)\n'
                 f'N={N_top}/{N_bottom}, θ={np.degrees(twist_angle):.0f}°, '
                 f'κ={kappa}, r0={r0} Å, n_exc={n_exc}', fontsize=11)
    plt.tight_layout()
    fname = f"EM_excitonic_sc{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved: {fname}")

    return omegas, results, Omega_trunc, A_trunc


def calculate_bse_absorbance(N_top=1, N_bottom=1, twist_angle=0.0,
                              E_range=(0.0, 1.0), n_E=500, eta=0.010,
                              k_range=0.15, n_k_bse=30,
                              n_val=2, n_cond=2,
                              kappa=2.5, r0=5.0,
                              plot_ipa_comparison=True,
                              band_window=None, save_prefix=""):
    r"""
    BSE excitonic optical absorbance spectrum.

    Computes:
        sigma^{bb}(ω) ∝ ω * Σ_S |d^b_S|² * δ(ω - Ω_S)

    where d^b_S = Σ_{vck} A^S_{vck} r^b_{cv}(k) is the exciton optical dipole.

    Parameters
    ----------
    n_k_bse : int
        k-grid per direction for BSE (total Nk = n_k_bse²).
    n_val, n_cond : int
        Number of valence/conduction bands in BSE active space.
    kappa : float
        Effective dielectric constant of environment.
    r0 : float
        2D polarizability screening length in Å.
    plot_ipa_comparison : bool
        If True, overlay IPA absorbance for comparison.
    """
    print("=" * 60)
    print("BSE Excitonic Absorbance Calculation (Effective Model)")
    print("=" * 60)
    print(f"  Grid: {n_k_bse}x{n_k_bse} = {n_k_bse**2} k-points")
    print(f"  Active space: {n_val}v x {n_cond}c")
    print(f"  BSE dimension: {n_val * n_cond * n_k_bse**2}")
    print(f"  Keldysh params: kappa={kappa}, r0={r0} Å")

    # 1. Model & k-grid
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    kx = np.linspace(-k_range, k_range, n_k_bse)
    ky = np.linspace(-k_range, k_range, n_k_bse)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    # 2. Single-particle solve
    print(f"\n[1] Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)
    Nb = evals.shape[1]

    # 3. Select active bands near the gap
    mid = Nb // 2
    if band_window is not None:
        v_idx = np.arange(band_window[0], band_window[1] + 1)
        c_idx = np.arange(band_window[2], band_window[3] + 1)
    else:
        v_idx = np.arange(mid - n_val, mid)
        c_idx = np.arange(mid, mid + n_cond)

    Nv = len(v_idx)
    Nc = len(c_idx)
    print(f"  Bands: valence {v_idx}, conduction {c_idx}")
    print(f"  QP gap range: {np.min(evals[:, c_idx[0]] - evals[:, v_idx[-1]]):.4f} - "
          f"{np.max(evals[:, c_idx[-1]] - evals[:, v_idx[0]]):.4f} eV")

    # 4. Velocity matrices in eigenbasis
    print(f"\n[2] Computing velocity matrices...")
    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))

    vx_eig = U_dag @ vx_orb @ U  # (Nk, Nb, Nb)
    vy_eig = U_dag @ vy_orb @ U

    # 5. Interband position matrix elements r^b_{cv}(k) = v^b_{cv} / (i * omega_{cv})
    E_v_arr = evals[:, v_idx]  # (Nk, Nv)
    E_c_arr = evals[:, c_idx]  # (Nk, Nc)
    dE = E_c_arr[:, None, :] - E_v_arr[:, :, None]  # (Nk, Nv, Nc)

    eps_denom = 1e-5
    v_eig_map = {'x': vx_eig, 'y': vy_eig}

    r_b = {}
    for b_dir in ['x', 'y']:
        v_b = v_eig_map[b_dir]
        vb_cv = v_b[:, c_idx, :][:, :, v_idx]  # (Nk, Nc, Nv)
        vb_cv = np.transpose(vb_cv, (0, 2, 1))  # (Nk, Nv, Nc)
        rb = np.zeros_like(vb_cv)
        valid = np.abs(dE) > eps_denom
        rb[valid] = vb_cv[valid] / (1j * dE[valid])
        r_b[b_dir] = rb  # (Nk, Nv, Nc)

    # 6. Build & diagonalize BSE Hamiltonian
    a_lat = model.a_lat
    b_lat = model.b_lat
    A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2
    print(f"  Moiré unit cell area: {A_uc:.1f} Å²")

    print(f"\n[3] Building BSE Hamiltonian...")
    H_bse = build_bse_hamiltonian(evals, evecs, k_points, v_idx, c_idx, A_uc,
                                   kappa=kappa, r0=r0, N_top=N_top, N_bottom=N_bottom)

    print(f"\n[4] Diagonalizing BSE ({H_bse.shape[0]}x{H_bse.shape[0]})...")
    import time
    t0 = time.time()
    dim_bse_mat = H_bse.shape[0]
    n_exciton_max = min(1000, dim_bse_mat - 2)
    if dim_bse_mat > 10000:
        print(f"    Using sparse eigsh (lowest {n_exciton_max} states)...")
        Omega_S, A_coeff = eigsh(H_bse, k=n_exciton_max, which='SM')
        sort_idx = np.argsort(Omega_S)
        Omega_S = Omega_S[sort_idx]
        A_coeff = A_coeff[:, sort_idx]
    else:
        Omega_S, A_coeff = scipy_eigh(H_bse, driver='evd')
    dt = time.time() - t0
    print(f"    Done in {dt:.1f} s")
    print(f"    Exciton energy range: {Omega_S[0]:.4f} - {Omega_S[-1]:.4f} eV")
    print(f"    Lowest exciton: {Omega_S[0]:.4f} eV  (QP gap ~ {np.min(dE):.4f} eV)")
    print(f"    Binding energy estimate: {np.min(dE) - Omega_S[0]:.4f} eV")
    print(f"    Next Lowest exciton: {Omega_S[1]:.4f} eV  (QP gap ~ {np.min(dE):.4f} eV)")
    print(f"    Binding energy estimate: {np.min(dE) - Omega_S[1]:.4f} eV")


    dim_bse = Nv * Nc * Nk

    # 7. Compute BSE absorbance
    print(f"\n[5] Computing BSE absorbance spectrum...")
    omegas = np.linspace(E_range[0], E_range[1], n_E)
    abs_bse = {}

    for b_dir in ['x', 'y']:
        r_b_flat = r_b[b_dir].reshape(dim_bse)
        d_b_S = A_coeff.conj().T @ r_b_flat  # (N_excitons,) complex
        osc_S = np.abs(d_b_S)**2  # (N_excitons,)

        # F-sum rule check: BSE vs IPA integrated oscillator strength must match
        bse_osc_sum = np.sum(osc_S)
        ipa_osc_sum = np.sum(np.abs(r_b_flat)**2)
        print(f"    F-sum check ({b_dir}-pol): BSE={bse_osc_sum:.6e}, IPA={ipa_osc_sum:.6e}, "
                f"ratio={bse_osc_sum/ipa_osc_sum:.6f}")

        # σ(ω) ∝ (1/Nk) Σ_S |d^b_S|² δ(ω - Ω_S)
        diff = omegas[:, None] - Omega_S[None, :]
        lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
        sigma = lorentz @ osc_S / Nk

        # absorbance ∝ ω × σ
        abs_bse[b_dir] = sigma * omegas

    # 8. Optionally compute IPA absorbance on the same grid
    abs_ipa = None
    if plot_ipa_comparison:
        print(f"\n[6] Computing IPA comparison on same grid...")
        abs_ipa = {}
        dE_flat = dE.reshape(dim_bse)

        for b_dir in ['x', 'y']:
            rb_sq = np.abs(r_b[b_dir].reshape(dim_bse))**2

            diff = omegas[:, None] - dE_flat[None, :]
            lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
            sigma_ipa = lorentz @ rb_sq / Nk

            abs_ipa[b_dir] = sigma_ipa * omegas

    # 9. Plotting
    plt.figure(figsize=(8, 6))

    plt.plot(omegas, abs_bse['x'], 'r-', lw=2, label=r'BSE x-pol')
    plt.plot(omegas, abs_bse['y'], 'b--', lw=2, label=r'BSE y-pol')
    if abs_ipa is not None:
        plt.plot(omegas, abs_ipa['x'], 'r-.', lw=1.5, alpha=0.5, label=r'IPA x-pol')
        plt.plot(omegas, abs_ipa['y'], 'b-.', lw=1.5, alpha=0.5, label=r'IPA y-pol')

    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Optical Absorbance (a.u.)')
    plt.title(f'BSE Absorbance (Effective Model)\n'
              f'N={N_top}/{N_bottom}, twist={np.degrees(twist_angle):.0f}°, '
              f'kappa={kappa}, r0={r0} Å, grid={n_k_bse}²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)

    fname = f"EM_bse_absorbance{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved BSE Absorbance Figure: {fname}")

    return omegas, abs_bse, Omega_S, A_coeff


def plot_exciton_oscillator_strength(N_top=1, N_bottom=1, twist_angle=0.0,
                                      E_range=(0.0, 1.0), eta=0.010,
                                      k_range=0.15, n_k_bse=30,
                                      n_val=2, n_cond=2,
                                      kappa=2.5, r0=5.0,
                                      polarization='x',
                                      n_show=100,
                                      plot_broadened=True,
                                      band_window=None, save_prefix=""):
    r"""
    Compute and plot exciton oscillator strength for a given light polarization.

    Solves the BSE and plots |d^b_S|^2 (oscillator strength) of each exciton
    state S as a stem plot versus exciton energy Omega_S.

    The exciton optical dipole is:
        d^b_S = \sum_{vck} A^S_{vck} r^b_{cv}(k)

    where r^b_{cv} = v^b_{cv} / (i omega_{cv}) is the interband position matrix
    element, and A^S are BSE eigenvector coefficients.

    Degenerate exciton multiplets are rotated into polarization eigenstates
    so that each stem is cleanly x-bright or y-bright.

    Parameters
    ----------
    polarization : str
        Light polarization direction: 'x', 'y', or 'both'.
    n_show : int
        Maximum number of excitons to display in the stem plot.
    plot_broadened : bool
        If True, overlay Lorentzian-broadened absorption envelope.
    eta : float
        Lorentzian broadening width (eV) for the envelope.
    """
    print("=" * 60)
    print("Exciton Oscillator Strength (Effective Model)")
    print("=" * 60)
    print(f"  Polarization: {polarization}")
    print(f"  Grid: {n_k_bse}x{n_k_bse} = {n_k_bse**2} k-points")
    print(f"  Active space: {n_val}v x {n_cond}c")
    print(f"  Keldysh params: kappa={kappa}, r0={r0} Å")

    # 1. Model & k-grid
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    kx = np.linspace(-k_range, k_range, n_k_bse)
    ky = np.linspace(-k_range, k_range, n_k_bse)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    # 2. Single-particle solve
    print(f"\n[1] Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)
    Nb = evals.shape[1]

    # 3. Select active bands
    mid = Nb // 2
    if band_window is not None:
        v_idx = np.arange(band_window[0], band_window[1] + 1)
        c_idx = np.arange(band_window[2], band_window[3] + 1)
    else:
        v_idx = np.arange(mid - n_val, mid)
        c_idx = np.arange(mid, mid + n_cond)

    Nv = len(v_idx)
    Nc = len(c_idx)
    print(f"  Bands: valence {v_idx}, conduction {c_idx}")
    qp_gap = np.min(evals[:, c_idx[0]] - evals[:, v_idx[-1]])
    print(f"  QP gap: {qp_gap:.4f} eV")

    # 4. Velocity matrices in eigenbasis
    print(f"\n[2] Computing velocity matrices...")
    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))

    vx_eig = U_dag @ vx_orb @ U
    vy_eig = U_dag @ vy_orb @ U

    # 5. Interband position matrix elements r^b_{cv}(k)
    E_v_arr = evals[:, v_idx]
    E_c_arr = evals[:, c_idx]
    dE = E_c_arr[:, None, :] - E_v_arr[:, :, None]  # (Nk, Nv, Nc)

    eps_denom = 1e-5
    v_eig_map = {'x': vx_eig, 'y': vy_eig}

    r_b = {}
    for b_dir in ['x', 'y']:
        v_b = v_eig_map[b_dir]
        vb_cv = v_b[:, c_idx, :][:, :, v_idx]
        vb_cv = np.transpose(vb_cv, (0, 2, 1))  # (Nk, Nv, Nc)
        rb = np.zeros_like(vb_cv)
        valid = np.abs(dE) > eps_denom
        rb[valid] = vb_cv[valid] / (1j * dE[valid])
        r_b[b_dir] = rb

    # 6. Build & diagonalize BSE Hamiltonian
    a_lat = model.a_lat
    b_lat = model.b_lat
    A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2

    print(f"\n[3] Building BSE Hamiltonian...")
    H_bse = build_bse_hamiltonian(evals, evecs, k_points, v_idx, c_idx, A_uc,
                                   kappa=kappa, r0=r0, N_top=N_top, N_bottom=N_bottom)

    print(f"\n[4] Diagonalizing BSE ({H_bse.shape[0]}x{H_bse.shape[0]})...")
    import time
    t0 = time.time()
    dim_bse_mat = H_bse.shape[0]
    n_exciton_max = min(1000, dim_bse_mat - 2)
    if dim_bse_mat > 10000:
        print(f"    Using sparse eigsh (lowest {n_exciton_max} states)...")
        Omega_S, A_coeff = eigsh(H_bse, k=n_exciton_max, which='SM')
        sort_idx = np.argsort(Omega_S)
        Omega_S = Omega_S[sort_idx]
        A_coeff = A_coeff[:, sort_idx]
    else:
        Omega_S, A_coeff = scipy_eigh(H_bse, driver='evd')
    dt = time.time() - t0
    print(f"    Done in {dt:.1f} s")
    print(f"    Exciton energies: {Omega_S[0]:.4f} - {Omega_S[-1]:.4f} eV")
    print(f"    Binding energy: {qp_gap - Omega_S[0]:.4f} eV")

    dim_bse = Nv * Nc * Nk

    # 7. Resolve degenerate excitons into polarization eigenstates
    r_b_x_flat = r_b['x'].reshape(dim_bse)
    r_b_y_flat = r_b['y'].reshape(dim_bse)
    A_coeff = _resolve_degenerate_excitons(Omega_S, A_coeff,
                                            r_b_x_flat, r_b_y_flat)

    # 8. Compute oscillator strength |d^b_S|^2 for each exciton
    osc = {}
    for b_dir in ['x', 'y']:
        r_flat = r_b[b_dir].reshape(dim_bse)
        d_S = A_coeff.conj().T @ r_flat  # (N_excitons,) complex
        osc[b_dir] = np.abs(d_S)**2 / Nk

    # 9. Select excitons in energy range
    mask = (Omega_S >= E_range[0]) & (Omega_S <= E_range[1])
    idx = np.where(mask)[0][:n_show]
    E_sel = Omega_S[idx]

    pol_list = ['x', 'y'] if polarization == 'both' else [polarization]

    for b_dir in pol_list:
        osc_sel = osc[b_dir][idx]

        # Print brightest excitons
        bright_order = np.argsort(osc_sel)[::-1]
        print(f"\n  Top 5 brightest excitons ({b_dir}-pol):")
        for rank, si in enumerate(bright_order[:5]):
            print(f"    #{rank+1}: E = {E_sel[si]:.4f} eV, "
                  f"|d^{b_dir}|² = {osc_sel[si]:.4e}")

    # 10. Plot
    if polarization == 'both':
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for ax, b_dir, color in zip(axes, ['x', 'y'], ['red', 'blue']):
            osc_sel = osc[b_dir][idx]
            markerline, stemlines, baseline = ax.stem(
                E_sel, osc_sel, linefmt='-', markerfmt='o', basefmt='k-')
            markerline.set_color(color)
            stemlines.set_color(color)

            if plot_broadened:
                n_E = 500
                omegas = np.linspace(E_range[0], E_range[1], n_E)
                diff = omegas[:, None] - Omega_S[None, :]
                lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
                envelope = lorentz @ osc[b_dir] / 1.0
                scale = np.max(osc_sel) / np.max(envelope) if np.max(envelope) > 0 else 1.0
                ax.plot(omegas, envelope * scale, color=color, alpha=0.3, lw=1.5)

            ax.set_ylabel(f'$|d^{b_dir}_S|^2$ (arb.)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(E_range)

        axes[0].set_title(f'Exciton Oscillator Strength\n'
                          f'N={N_top}/{N_bottom}, twist={np.degrees(twist_angle):.0f}°, '
                          f'κ={kappa}, r0={r0} Å, grid={n_k_bse}²')
        axes[-1].set_xlabel('Exciton Energy (eV)')
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        b_dir = polarization
        color = 'red' if b_dir == 'x' else 'blue'
        osc_sel = osc[b_dir][idx]

        markerline, stemlines, baseline = ax.stem(
            E_sel, osc_sel, linefmt='-', markerfmt='o', basefmt='k-')
        markerline.set_color(color)
        stemlines.set_color(color)

        if plot_broadened:
            n_E = 500
            omegas = np.linspace(E_range[0], E_range[1], n_E)
            diff = omegas[:, None] - Omega_S[None, :]
            lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
            envelope = lorentz @ osc[b_dir] / 1.0
            scale = np.max(osc_sel) / np.max(envelope) if np.max(envelope) > 0 else 1.0
            ax.plot(omegas, envelope * scale, color=color, alpha=0.3, lw=1.5)

        ax.set_xlabel('Exciton Energy (eV)')
        ax.set_ylabel(f'$|d^{b_dir}_S|^2$ (arb.)')
        ax.set_title(f'Exciton Oscillator Strength ({b_dir}-polarized)\n'
                     f'N={N_top}/{N_bottom}, twist={np.degrees(twist_angle):.0f}°, '
                     f'κ={kappa}, r0={r0} Å, grid={n_k_bse}²')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(E_range)

    plt.tight_layout()
    fname = f"EM_exciton_osc_strength_{polarization}{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved: {fname}")

    return Omega_S, osc


def plot_exciton_analysis(Omega_S, osc_data, shift_weight_data, E_range=(0.0, 1.0),
                          N_top=1, N_bottom=1, twist_angle=0.0,
                          kappa=2.5, r0=5.0, thickness=5.2,
                          n_show=50, save_prefix=""):
    """
    Diagnostic stem plot of exciton properties.

    Parameters
    ----------
    Omega_S : ndarray
        Exciton energies.
    osc_data : dict
        {'x': |d^x_S|², 'y': |d^y_S|²} oscillator strengths.
    shift_weight_data : dict
        {'x': Re[d^{x*}_S g^{xz}_S], 'y': ...} gauge-invariant shift weights.
    n_show : int
        Number of lowest excitons to display.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Select excitons in energy range
    mask = (Omega_S >= E_range[0]) & (Omega_S <= E_range[1])
    idx = np.where(mask)[0][:n_show]
    E_sel = Omega_S[idx]

    # Top panel: oscillator strength
    ax = axes[0]
    for b_dir, color, label in [('x', 'red', '$|d^x_S|^2$'), ('y', 'blue', '$|d^y_S|^2$')]:
        osc_sel = osc_data[b_dir][idx]
        ax.stem(E_sel, osc_sel, linefmt=f'{color[0]}-', markerfmt=f'{color[0]}o',
                basefmt='k-', label=label)
    ax.set_ylabel('Oscillator Strength (arb.)')
    ax.legend()
    ax.set_title(f'Exciton Analysis (N={N_top}/{N_bottom}, '
                 f'twist={np.degrees(twist_angle):.0f}°, '
                 f'kappa={kappa}, r0={r0} Å)')
    ax.grid(True, alpha=0.3)

    # Bottom panel: shift weight Re[d* g^{bz}] for each polarization
    ax = axes[1]
    for b_dir, color, label in [('x', 'red', r'$\mathrm{Re}[d^{x*} g^{xz}]$'),
                                 ('y', 'blue', r'$\mathrm{Re}[d^{y*} g^{yz}]$')]:
        sw_sel = shift_weight_data[b_dir][idx]
        pos = sw_sel >= 0
        if np.any(pos):
            ax.stem(E_sel[pos], sw_sel[pos], linefmt=f'{color[0]}-', markerfmt=f'{color[0]}o',
                    basefmt='k-', label=f'{label} > 0')
        if np.any(~pos):
            ax.stem(E_sel[~pos], sw_sel[~pos], linefmt=f'{color[0]}--', markerfmt=f'{color[0]}s',
                    basefmt='k-', label=f'{label} < 0')

    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Exciton Energy (eV)')
    ax.set_ylabel(r'Shift Weight (arb.)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(E_range)

    plt.tight_layout()
    fname = f"EM_bse_exciton_analysis{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved Exciton Analysis Figure: {fname}")


def _resolve_degenerate_excitons(Omega_S, A_coeff, r_b_x_flat, r_b_y_flat,
                                  degen_tol=1e-3):
    """
    Rotate degenerate exciton subspaces so that each state is an eigenstate
    of polarization (x-bright or y-bright), rather than an arbitrary mixture.

    Within each degenerate multiplet, diagonalizes the x-polarization
    oscillator-strength matrix  D^x_{ij} = d^x_i (d^x_j)*  to find the
    linear combination that maximally couples to x-polarized light.
    The orthogonal complement then couples to y.

    Parameters
    ----------
    degen_tol : float
        Energy tolerance (eV) for grouping excitons as degenerate.

    Returns
    -------
    A_coeff_rot : ndarray
        Rotated BSE eigenvectors (same shape as A_coeff).
    """
    dim_bse = A_coeff.shape[0]
    A_rot = A_coeff.copy()

    # Group into degenerate multiplets
    i = 0
    while i < len(Omega_S):
        j = i + 1
        while j < len(Omega_S) and abs(Omega_S[j] - Omega_S[i]) < degen_tol:
            j += 1
        deg = j - i  # multiplicity
        if deg > 1:
            idx = slice(i, j)
            # x-dipoles within this multiplet: d^x_m = A[:,m]^dagger @ r_x
            d_x = A_rot[:, idx].conj().T @ r_b_x_flat  # (deg,) complex
            # Oscillator strength matrix for x-pol: D_ij = d_i d_j*
            D_x = np.outer(d_x, d_x.conj())  # (deg, deg)
            # Diagonalize — eigenstates are the polarization-resolved excitons
            _, U_rot = np.linalg.eigh(D_x)
            # Rotate BSE coefficients: new = old @ U_rot
            A_rot[:, idx] = A_rot[:, idx] @ U_rot
        i = j

    return A_rot


def analyze_exciton_wavefunction(N_top=1, N_bottom=1, twist_angle=0.0,
                                  E_range=(0.0, 1.0), eta=0.010,
                                  k_range=0.15, n_k_bse=30,
                                  n_val=2, n_cond=2,
                                  thickness=5.2,
                                  kappa=2.5, r0=5.0,
                                  n_excitons=4,
                                  band_window=None, save_prefix=""):
    r"""
    Analyze the composition and real-space envelope of the lowest bright excitons.

    Degenerate exciton multiplets are rotated to the polarization basis
    (x-bright vs y-bright) before analysis, so the layer composition
    reflects the physical states excited by linearly polarized light.

    For each selected exciton S, produces:
      1. k-space weight map:  w(k) = Σ_{vc} |A^S_{vck}|²
      2. Real-space envelope: |Φ_S(r)|² = Σ_{vc} |FFT[A^S_{vc}(k)]|²
         (probability distribution of electron-hole relative coordinate)
      3. Layer-resolved electron and hole densities:
         ρ_e(layer) = Σ_{ck} |u_c^layer(k)|² Σ_v |A^S_{vck}|²
         ρ_h(layer) = Σ_{vk} |u_v^layer(k)|² Σ_c |A^S_{vck}|²

    Parameters
    ----------
    n_excitons : int
        Number of brightest excitons to analyze.
    """
    print("=" * 60)
    print("Exciton Wavefunction Analysis")
    print("=" * 60)

    # 1. Model & k-grid
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    kx = np.linspace(-k_range, k_range, n_k_bse)
    ky = np.linspace(-k_range, k_range, n_k_bse)
    dk = kx[1] - kx[0]
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    # 2. Single-particle solve
    print(f"[1] Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)
    Nb = evals.shape[1]

    mid = Nb // 2
    if band_window is not None:
        v_idx = np.arange(band_window[0], band_window[1] + 1)
        c_idx = np.arange(band_window[2], band_window[3] + 1)
    else:
        v_idx = np.arange(mid - n_val, mid)
        c_idx = np.arange(mid, mid + n_cond)
    Nv = len(v_idx)
    Nc = len(c_idx)

    # 3. Velocity for dipole matrix elements
    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))
    vx_eig = U_dag @ vx_orb @ U
    vy_eig = U_dag @ vy_orb @ U

    E_v_arr = evals[:, v_idx]
    E_c_arr = evals[:, c_idx]
    dE = E_c_arr[:, None, :] - E_v_arr[:, :, None]

    eps_denom = 1e-5
    r_b = {}
    for b_dir, v_b in [('x', vx_eig), ('y', vy_eig)]:
        vb_cv = np.transpose(v_b[:, c_idx, :][:, :, v_idx], (0, 2, 1))
        rb = np.zeros_like(vb_cv)
        valid = np.abs(dE) > eps_denom
        rb[valid] = vb_cv[valid] / (1j * dE[valid])
        r_b[b_dir] = rb

    # 4. BSE
    a_lat = model.a_lat
    b_lat = model.b_lat
    A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2

    print(f"[2] Building & diagonalizing BSE...")
    H_bse = build_bse_hamiltonian(evals, evecs, k_points, v_idx, c_idx, A_uc,
                                   kappa=kappa, r0=r0, N_top=N_top, N_bottom=N_bottom)
    dim_bse_mat = H_bse.shape[0]
    n_exciton_max = min(1000, dim_bse_mat - 2)
    if dim_bse_mat > 10000:
        print(f"    Using sparse eigsh (lowest {n_exciton_max} states)...")
        Omega_S, A_coeff = eigsh(H_bse, k=n_exciton_max, which='SM')
        sort_idx = np.argsort(Omega_S)
        Omega_S = Omega_S[sort_idx]
        A_coeff = A_coeff[:, sort_idx]
    else:
        Omega_S, A_coeff = scipy_eigh(H_bse, driver='evd')
    del H_bse
    dim_bse = Nv * Nc * Nk

    print(f"    Lowest exciton: {Omega_S[0]:.4f} eV")
    print(f"    Binding energy: {np.min(dE) - Omega_S[0]:.4f} eV")

    # 5. Resolve degenerate subspaces into x-bright / y-bright eigenstates
    r_x_flat = r_b['x'].reshape(dim_bse)
    r_y_flat = r_b['y'].reshape(dim_bse)

    print(f"\n[3] Resolving degenerate exciton subspaces by polarization...")
    A_coeff = _resolve_degenerate_excitons(Omega_S, A_coeff, r_x_flat, r_y_flat)

    # Recompute dipoles with rotated coefficients
    d_x_all = A_coeff.conj().T @ r_x_flat  # (N_excitons,)
    d_y_all = A_coeff.conj().T @ r_y_flat
    osc_x = np.abs(d_x_all)**2
    osc_y = np.abs(d_y_all)**2
    osc_total = osc_x + osc_y

    # Filter to energy range and pick brightest
    in_range = (Omega_S >= E_range[0]) & (Omega_S <= E_range[1])
    osc_masked = np.where(in_range, osc_total, 0.0)
    bright_idx = np.argsort(osc_masked)[::-1][:n_excitons]
    bright_idx = np.sort(bright_idx)  # sort by energy

    print(f"\n[4] {n_excitons} brightest excitons (polarization-resolved):")
    for i, si in enumerate(bright_idx):
        pol = 'x-bright' if osc_x[si] > osc_y[si] else 'y-bright'
        print(f"    S{i}: E = {Omega_S[si]:.4f} eV, "
              f"|d_x|² = {osc_x[si]:.4e}, |d_y|² = {osc_y[si]:.4e}  ({pol})")

    # 6. Compute properties for each selected exciton
    # Layer labels for the 4 orbitals (sublattice basis): [top_A, top_B, bot_A, bot_B]
    layer_labels = ['Top A', 'Top B', 'Bot A', 'Bot B']

    # Real-space grid from FFT
    rx = np.fft.fftshift(np.fft.fftfreq(n_k_bse, d=dk/(2*np.pi)))
    ry = rx.copy()
    RX, RY = np.meshgrid(rx, ry)

    # === FIGURE: one row per exciton, 3 columns ===
    n_show = len(bright_idx)
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4.2 * n_show),
                              squeeze=False)

    for row, si in enumerate(bright_idx):
        A_S = A_coeff[:, si]  # (dim_bse,)
        pol = 'x-bright' if osc_x[si] > osc_y[si] else 'y-bright'

        # Reshape to (Nk, Nv, Nc) — BSE index: I = k*(Nv*Nc) + v*Nc + c
        A_3d = A_S.reshape(Nk, Nv, Nc)
        # Reshape to (n_ky, n_kx, Nv, Nc) matching meshgrid order
        A_4d = A_3d.reshape(n_k_bse, n_k_bse, Nv, Nc)

        # --- Column 0: k-space weight map ---
        wk = np.sum(np.abs(A_4d)**2, axis=(2, 3))  # (n_ky, n_kx)
        ax = axes[row, 0]
        im = ax.pcolormesh(KX, KY, wk, cmap='hot', shading='auto')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (Å$^{-1}$)')
        ax.set_title(f'S{row} ({pol}): $E$ = {Omega_S[si]:.4f} eV\n'
                     r'$\sum_{vc}|A^S_{vck}|^2$')
        ax.set_aspect('equal')

        # --- Column 1: Real-space envelope |Φ(r)|² ---
        psi_r_sq = np.zeros((n_k_bse, n_k_bse))
        for iv in range(Nv):
            for ic in range(Nc):
                phi_k = A_4d[:, :, iv, ic]
                phi_r = np.fft.fftshift(np.fft.fft2(phi_k))
                psi_r_sq += np.abs(phi_r)**2
        psi_r_sq /= np.max(psi_r_sq)

        ax = axes[row, 1]
        im = ax.pcolormesh(RX, RY, psi_r_sq, cmap='inferno', shading='auto')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel(r'$\Delta x$ (Å)')
        ax.set_ylabel(r'$\Delta y$ (Å)')
        ax.set_title(f'S{row} ({pol}): Real-space envelope\n'
                     r'$|\Phi^S(\mathbf{r}_e - \mathbf{r}_h)|^2$')
        ax.set_aspect('equal')

        # --- Column 2: Layer-resolved electron & hole density ---
        w_e_ck = np.sum(np.abs(A_3d)**2, axis=1)  # (Nk, Nc) — sum over v
        w_h_vk = np.sum(np.abs(A_3d)**2, axis=2)  # (Nk, Nv) — sum over c

        U_c = evecs[:, :, c_idx]  # (Nk, dim_H, Nc)
        U_v = evecs[:, :, v_idx]  # (Nk, dim_H, Nv)

        rho_e = np.zeros(Nb)
        rho_h = np.zeros(Nb)
        for ic in range(Nc):
            uc_sq = np.abs(U_c[:, :, ic])**2  # (Nk, dim_H)
            rho_e += np.sum(uc_sq * w_e_ck[:, ic:ic+1], axis=0)
        for iv in range(Nv):
            uv_sq = np.abs(U_v[:, :, iv])**2  # (Nk, dim_H)
            rho_h += np.sum(uv_sq * w_h_vk[:, iv:iv+1], axis=0)

        rho_e /= np.sum(rho_e)
        rho_h /= np.sum(rho_h)

        # Group into Top / Bottom for cleaner display
        rho_e_top = rho_e[0] + rho_e[1]
        rho_e_bot = rho_e[2] + rho_e[3] if Nb == 4 else 0.0
        rho_h_top = rho_h[0] + rho_h[1]
        rho_h_bot = rho_h[2] + rho_h[3] if Nb == 4 else 0.0

        ax = axes[row, 2]
        x_pos = np.arange(Nb)
        width = 0.35
        bars_e = ax.bar(x_pos - width/2, rho_e, width, color='#e41a1c', label='Electron')
        bars_h = ax.bar(x_pos + width/2, rho_h, width, color='#377eb8', label='Hole')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layer_labels[:Nb], rotation=30, ha='right')
        ax.set_ylabel('Weight')
        ax.set_title(f'S{row} ({pol}): Layer composition\n'
                     f'e: Top {rho_e_top:.0%} Bot {rho_e_bot:.0%} | '
                     f'h: Top {rho_h_top:.0%} Bot {rho_h_bot:.0%}')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in list(zip(bars_e, rho_e)) + list(zip(bars_h, rho_h)):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.0%}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'Exciton Wavefunction Analysis (polarization-resolved)\n'
                 f'N={N_top}/{N_bottom}, twist={np.degrees(twist_angle):.0f}°, '
                 f'kappa={kappa}, r0={r0} Å, grid={n_k_bse}²', fontsize=12)
    plt.tight_layout()
    fname = f"EM_exciton_wavefunctions{save_prefix}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fname}")

    return Omega_S, A_coeff, bright_idx


def plot_exciton_level(N_top=1, N_bottom=[2,7], twist_angle=0.0,
                                      E_range=(0.0, 1.0),
                                      k_range=0.15, n_k_bse=30,
                                      n_val=2, n_cond=2,
                                      kappa=2.5, r0=5.0,
                                      band_window=None,
                                      E_g=2.1, gamma_c = 0.58, gamma_v = -0.32,):
    bright_level = []
    for N_bot in range(N_bottom[0], N_bottom[1]+1):
        # 1. Model & k-grid
        model = TwistedBPModel(N_top=N_top, N_bottom=N_bot, twist_angle=twist_angle)

        kx = np.linspace(-k_range, k_range, n_k_bse)
        ky = np.linspace(-k_range, k_range, n_k_bse)
        KX, KY = np.meshgrid(kx, ky)
        k_points = np.column_stack([KX.flatten(), KY.flatten()])
        Nk = len(k_points)

        # 2. Single-particle solve
        print(f"\n[1] Diagonalizing H for {Nk} k-points...")
        H_stack = model.get_hamiltonians(k_points)
        evals, evecs = np.linalg.eigh(H_stack)
        Nb = evals.shape[1]

        # 3. Select active bands
        mid = Nb // 2
        if band_window is not None:
            v_idx = np.arange(band_window[0], band_window[1] + 1)
            c_idx = np.arange(band_window[2], band_window[3] + 1)
        else:
            v_idx = np.arange(mid - n_val, mid)
            c_idx = np.arange(mid, mid + n_cond)

        Nv = len(v_idx)
        Nc = len(c_idx)
        print(f"  Bands: valence {v_idx}, conduction {c_idx}")
        qp_gap = np.min(evals[:, c_idx[0]] - evals[:, v_idx[-1]])
        print(f"  QP gap: {qp_gap:.4f} eV")

        # 4. Velocity matrices in eigenbasis
        print(f"\n[2] Computing velocity matrices...")
        vx_orb, vy_orb = model.get_velocity_matrices(k_points)
        U = evecs
        U_dag = np.conj(np.transpose(U, (0, 2, 1)))

        vx_eig = U_dag @ vx_orb @ U
        vy_eig = U_dag @ vy_orb @ U

        # 5. Interband position matrix elements r^b_{cv}(k)
        E_v_arr = evals[:, v_idx]
        E_c_arr = evals[:, c_idx]
        dE = E_c_arr[:, None, :] - E_v_arr[:, :, None]  # (Nk, Nv, Nc)

        eps_denom = 1e-5
        v_eig_map = {'x': vx_eig, 'y': vy_eig}

        r_b = {}
        for b_dir in ['x', 'y']:
            v_b = v_eig_map[b_dir]
            vb_cv = v_b[:, c_idx, :][:, :, v_idx]
            vb_cv = np.transpose(vb_cv, (0, 2, 1))  # (Nk, Nv, Nc)
            rb = np.zeros_like(vb_cv)
            valid = np.abs(dE) > eps_denom
            rb[valid] = vb_cv[valid] / (1j * dE[valid])
            r_b[b_dir] = rb

        # 6. Build & diagonalize BSE Hamiltonian
        a_lat = model.a_lat
        b_lat = model.b_lat
        A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2

        print(f"\n[3] Building BSE Hamiltonian...")
        H_bse = build_bse_hamiltonian(evals, evecs, k_points, v_idx, c_idx, A_uc,
                                    kappa=kappa, r0=r0, N_top=N_top, N_bottom=N_bot)

        print(f"\n[4] Diagonalizing BSE ({H_bse.shape[0]}x{H_bse.shape[0]})...")
        import time
        t0 = time.time()
        dim_bse_mat = H_bse.shape[0]
        n_exciton_max = min(1000, dim_bse_mat - 2)
        if dim_bse_mat > 10000:
            print(f"    Using sparse eigsh (lowest {n_exciton_max} states)...")
            Omega_S, A_coeff = eigsh(H_bse, k=n_exciton_max, which='SM')
            sort_idx = np.argsort(Omega_S)
            Omega_S = Omega_S[sort_idx]
            A_coeff = A_coeff[:, sort_idx]
        else:
            Omega_S, A_coeff = scipy_eigh(H_bse, driver='evd')
        dt = time.time() - t0
        print(f"    Done in {dt:.1f} s")
        print(f"    Exciton energies: {Omega_S[0]:.4f} - {Omega_S[-1]:.4f} eV")
        print(f"    Binding energy: {qp_gap - Omega_S[0]:.4f} eV")

        dim_bse = Nv * Nc * Nk

        # 7. Resolve degenerate excitons into polarization eigenstates
        r_b_x_flat = r_b['x'].reshape(dim_bse)
        r_b_y_flat = r_b['y'].reshape(dim_bse)
        A_coeff = _resolve_degenerate_excitons(Omega_S, A_coeff,
                                                r_b_x_flat, r_b_y_flat)

        # 8. Compute oscillator strength |d^b_S|^2 for each exciton
        osc = {}
        for b_dir in ['x', 'y']:
            r_flat = r_b[b_dir].reshape(dim_bse)
            d_S = A_coeff.conj().T @ r_flat  # (N_excitons,) complex
            osc[b_dir] = np.abs(d_S)**2 / Nk

        # 9. Select excitons in energy range
        mask = (Omega_S >= E_range[0]) & (Omega_S <= E_range[1])
        idx = np.where(mask)[0]
        E_sel = Omega_S[idx]

        pol_list = ['x', 'y']
        bright_exciton = []

        for b_dir in pol_list:
            osc_sel = osc[b_dir][idx]

            # Print brightest excitons
            bright_order = np.argsort(osc_sel)[::-1]
            print(f"\n  Top 5 brightest excitons ({b_dir}-pol):")
            for rank, si in enumerate(bright_order[:5]):
                print(f"    #{rank+1}: E = {E_sel[si]:.4f} eV, "
                    f"|d^{b_dir}|² = {osc_sel[si]:.4e}")
            bright_exciton.append(E_sel[bright_order[0]])
        bright_level.append(bright_exciton)

    # 10. Get the analytic levels, ref: Huang et al., Science 386, 526–531 (2024)
    level_list = np.linspace(N_bottom[0], N_bottom[1]+2, 500)
    Y_bright = E_g - 2 * gamma_c * np.cos(np.pi/(level_list + N_top + 1)) + 2 * gamma_v * np.cos(np.pi/(N_top + 1))
    X_bright = E_g - 2 * gamma_c * np.cos(np.pi/(level_list + N_top + 1)) + 2 * gamma_v * np.cos(np.pi/(level_list + 1))
    bright_level = np.array(bright_level)

    # 11. Plotting
    plt.figure(figsize=(5, 5))
    plt.plot(level_list, X_bright, label='X-bright (analytic)', color='red', ls='--')
    plt.plot(level_list, Y_bright, label='Y-bright (analytic)', color='blue', ls='--')
    plt.scatter(range(N_bottom[0], N_bottom[1]+1), bright_level[:, 0], label='X-bright (BSE)', color='red', marker='o')
    plt.scatter(range(N_bottom[0], N_bottom[1]+1), bright_level[:, 1], label='Y-bright (BSE)', color='blue', marker='s')
    # plt.tight_layout()
    plt.ylim(0.2,1.6)
    plt.legend()
    plt.xlabel('N_bottom')
    plt.ylabel('Bright Exciton Energy (eV)')
    fname = f"EM_exciton_level.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved: {fname}")

if __name__ == "__main__":

    # BP parameters
    b_lat = 4.588
    a_lat = 3.296
    G_moire = 2 * np.pi * np.abs(1/b_lat - 1/a_lat)
    n_top = 3
    n_bottom = 3
    twist_angle = np.pi / 2
    kappa=5.0
    r0=6.0
    # twist_angle = 0.0
    erange = (0.0, 0.8)

    # single k point test
    # --------------------------------------------
    # model = TwistedBPModel(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle)
    # k_test = np.array([[0.0, 0.0]])
    # H_test = model.get_hamiltonians(k_test)
    # print(H_test[0])
    # print("Eigenvalues at Gamma:", np.linalg.eigvalsh(H_test[0]))

    # # Band structure
    # # # --------------------------------------------
    # cal_bands(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #           k_fine_steps=200, y_lim=(-0.5,1.0))

    # # # 3D Band Structure
    # # # --------------------------------------------
    # # plot_3d_bands(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    # #               k_range=G_moire/2, n_grid=60,
    # #               view_elev=15, view_azim=45)
                  
    # # # Optical Conductivity
    # # # --------------------------------------------
    # calculate_optical_conductivity(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                n_k=240, n_E=500,eta=0.010,
    #                                k_range=G_moire/2, E_range=erange)
                                   
    # # # Matrix Element Map (VBM -> CBM)
    # # # --------------------------------------------
    # plot_transition_matrix_elements(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                 band_indices=(1, 2),
    #                                 k_range=G_moire/2, n_k=160)
    
    # # # Bandgap scaling with N_bottom
    # # # --------------------------------------------
    # plot_bandgap_scaling(N_top=n_top, N_bottom=[n_top,10], twist_angle=twist_angle,
    #                         E_g=2.1, gamma_c = 0.58, gamma_v = -0.32,)


    # # # Shift Current Calculation
    # # # --------------------------------------------   
    # calculate_shift_current(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle, 
    #                         n_k=240, n_E=100,
    #                         E_range=erange, k_range=G_moire/2,)

    # # # Z-direction (out-of-plane) Shift Current
    # # # --------------------------------------------
    # calculate_z_shift_current(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                           n_k=240, n_E=250,
    #                         #   band_window=[0,1,2,2],
    #                           E_range=erange, k_range=G_moire/2)

    # # # BSE Excitonic Z-Shift Current
    # # # --------------------------------------------
    # calculate_bse_z_shift_current(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                n_k_bse=30, n_val=2, n_cond=2,
    #                                E_range=erange, k_range=G_moire/2,
    #                                kappa=kappa, r0=r0,)

    # # # Exciton Oscillator Strength (stem plot)
    # # # --------------------------------------------
    # plot_exciton_oscillator_strength(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                   E_range=erange, eta=0.010,
    #                                   k_range=G_moire/2, n_k_bse=30,
    #                                   n_val=2, n_cond=2,
    #                                   kappa=kappa, r0=r0,
    #                                   polarization='both')

    # # # BSE Excitonic Absorbance
    # # # --------------------------------------------
    # calculate_bse_absorbance(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                           E_range=erange, n_E=500, eta=0.010,
    #                           k_range=G_moire/2, n_k_bse=30,
    #                           n_val=2, n_cond=2,
    #                           kappa=kappa, r0=r0,
    #                           plot_ipa_comparison=True,)
    
    # # # Excitonic Absorbance
    # # # --------------------------------------------
    # analyze_exciton_wavefunction(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                 E_range=erange, eta=0.010,
    #                                 k_range=G_moire/2, n_k_bse=30,
    #                                 n_val=2, n_cond=2,
    #                                 thickness=5.2,
    #                                 kappa=kappa, r0=r0,
    #                                 n_excitons=4,
    #                                 )

    # # Excitonic Levels
    # # --------------------------------------------
    # plot_exciton_level(N_top=n_top, N_bottom=[n_bottom,9], twist_angle=twist_angle,
    #                                   E_range=erange,
    #                                   k_range=G_moire/2, n_k_bse=30,
    #                                   n_val=2, n_cond=2,
    #                                   kappa=kappa, r0=r0,
    #                                   E_g=2.1, gamma_c = 0.58, gamma_v = -0.32,)

    # # Full Excitonic Shift Current (Eq. 10)
    # # --------------------------------------------
    calculate_excitonic_shift_current(
        N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
        E_range=erange, n_E=200, eta=0.010,
        k_range=G_moire/2, n_k_bse=30,
        n_val=2, n_cond=2,
        kappa=kappa, r0=r0,
        n_exciton_keep=200,
        plot_ipa_comparison=True,
        band_window = [0, 1, 2, 3]
    )