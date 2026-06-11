import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import eigsh
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
mpl.rcParams['font.family'] = 'Arial'

try:
    import torch
except ImportError:
    torch = None


def _save_dat(fname, data, header, fmt="%.10e"):
    """Save numeric data next to the plotted figure for later replotting."""
    np.savetxt(fname, data, header=header, fmt=fmt)
    print(f"Saved data: {fname}")


def kramers_kronig_eps1_from_eps2(omega, eps2, eps_inf=1.0):
    r"""
    Compute the real dielectric function from epsilon_2 by the KK relation.

        epsilon_1(omega) = eps_inf + 2/pi * P int_0^inf
            Omega * epsilon_2(Omega) / (Omega^2 - omega^2) dOmega

    The integral is evaluated on the provided positive-energy grid using
    trapezoidal weights. The singular grid point is omitted as the numerical
    principal-value prescription.
    """
    omega = np.asarray(omega, dtype=np.float64)
    eps2 = np.asarray(eps2, dtype=np.float64)

    if omega.ndim != 1 or eps2.ndim != 1:
        raise ValueError("omega and eps2 must be one-dimensional arrays.")
    if omega.shape != eps2.shape:
        raise ValueError("omega and eps2 must have the same shape.")
    if len(omega) < 2:
        raise ValueError("At least two omega points are required for KK transform.")
    if np.any(np.diff(omega) <= 0.0):
        raise ValueError("omega grid must be strictly increasing.")
    if omega[0] < 0.0:
        raise ValueError("omega grid must be non-negative.")

    weights = np.empty_like(omega)
    weights[0] = 0.5 * (omega[1] - omega[0])
    weights[-1] = 0.5 * (omega[-1] - omega[-2])
    if len(omega) > 2:
        weights[1:-1] = 0.5 * (omega[2:] - omega[:-2])

    denom = omega[None, :]**2 - omega[:, None]**2
    numer = weights[None, :] * omega[None, :] * eps2[None, :]
    mask = ~np.isclose(denom, 0.0, rtol=0.0, atol=1e-14)
    integrand = np.zeros_like(denom)
    np.divide(numer, denom, out=integrand, where=mask)
    return eps_inf + (2.0 / np.pi) * np.sum(integrand, axis=1)


def dielectric_to_refractive_index(eps1, eps2):
    """Return n and kappa from epsilon = eps1 + i eps2."""
    eps_abs = np.sqrt(eps1**2 + eps2**2)
    n = np.sqrt(np.maximum((eps_abs + eps1) / 2.0, 0.0))
    kappa = np.sqrt(np.maximum((eps_abs - eps1) / 2.0, 0.0))
    return n, kappa


def normal_incidence_reflectivity(eps1, eps2):
    """Reflectivity for vacuum -> material at normal incidence."""
    n, kappa = dielectric_to_refractive_index(eps1, eps2)
    return ((n - 1.0)**2 + kappa**2) / ((n + 1.0)**2 + kappa**2)


def shift_conductivity_to_current(sigma_uA_per_V2, reflectivity,
                                  total_thickness_A, sample_width_um,
                                  intensity_W_cm2=1.6e4):
    r"""
    Convert shift-current conductivity to measured current.

        I = (1 - R) * sigma * 2 * I_light * d * w / (epsilon0 * c)

    Parameters use convenient lab units:
    sigma in microampere/V^2, thickness in Angstrom, width in micron, and
    optical intensity in W/cm^2. The returned current is in ampere.
    """
    eps0 = 8.8541878128e-12
    c_light = 2.99792458e8

    sigma_A_per_V2 = np.asarray(sigma_uA_per_V2, dtype=np.float64) * 1.0e-6
    reflectivity = np.asarray(reflectivity, dtype=np.float64)
    intensity_W_m2 = intensity_W_cm2 * 1.0e4
    total_thickness_m = total_thickness_A * 1.0e-10
    sample_width_m = sample_width_um * 1.0e-6

    field_sq = (1.0 - reflectivity) * 2.0 * intensity_W_m2 / (eps0 * c_light)
    return sigma_A_per_V2 * field_sq * total_thickness_m * sample_width_m


# ============================================================
# Two-band model of twisted multilayer black phosphorus
# ============================================================

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
        # bulk BP parameters from MP
        self.b_lat=4.544 # armchair direction
        self.a_lat=3.295 # zig-zag direction

        self.N_top = N_top
        self.N_bottom = N_bottom
        self.twist_angle = twist_angle

        # effective interface coupling strength (in eV)
        if N_top == 4:
            self.coupling = 0.075 # For 4+4, 0.075 is a good fit.
        elif N_top == 1:
            self.coupling = 0.120 # For 1+1, 0.120 is a good fit.
        elif N_top == 3:
            self.coupling = 0.095 # For 3+3, 0.095 is a good fit.
        elif N_top == 2:
            self.coupling = 0.095 # For 2+2, 0.095 is a good fit.

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
        # adjusted parameters to better fit the DFT band structure
        dt = 0.160
        self.t1p=0.524 - dt
        self.t2p=-0.200 if N_top == 2 else -0.150
        self.t3p=0.060
        self.t4p=-0.168 + dt
        # original parameters from Rudenko's paper
        # self.t1p=0.524
        # self.t2p=0.180
        # self.t3p=-0.123
        # self.t4p=-0.168

        # self.t5p=0.005 # we dont use this here

        # Precomputed geometric factors (k-independent, used in all H/v/w methods)
        sx = self.a1 * np.sin(self.alpha1 / 2)
        cx = self.a1 * np.cos(self.alpha1 / 2)
        cy = self.a2 * np.cos(self.beta)
        self._p1  = 2 * sx
        self._p2  = sx
        self._p3  = cx
        self._p4  = cx + 2 * cy
        self._p5  = cy
        self._p6  = 2 * cx + cy
        self._p8  = cx + cy
        self._p9  = 3 * sx
        self._p10 = 2 * sx + cy
        self._q1  = 2 * cx + 2 * cy
        self._q2  = sx + cy
        self._pf_top = np.cos(np.pi * self.N_top / (self.N_top + 1))
        self._pf_bot = np.cos(np.pi * self.N_bottom / (self.N_bottom + 1)) if self.N_bottom > 0 else 0.0

    def basic_block(self, k_points, twist_angle):
        """Compute the basic Hamiltonian for given k-points."""
        k_points = np.atleast_2d(k_points)
        kx_tmp, ky_tmp = k_points[:, 0], k_points[:, 1]
        kx = kx_tmp * np.cos(twist_angle) - ky_tmp * np.sin(twist_angle)
        ky = kx_tmp * np.sin(twist_angle) + ky_tmp * np.cos(twist_angle)

        # Intralayer terms
        tAA = (2 * self.t3 * np.cos(self._p1 * kx)
               + 2 * self.t7 * np.cos(self._q1 * ky)
               + 4 * self.t10 * np.cos(self._p1 * kx) * np.cos(self._q1 * ky))

        tAB = (2 * self.t1 * np.cos(self._p2 * kx) * np.exp(-1j * self._p3 * ky)
               + 2 * self.t4 * np.cos(self._p2 * kx) * np.exp(1j * self._p4 * ky)
               + 2 * self.t8 * np.cos(self._p9 * kx) * np.exp(-1j * self._p3 * ky))

        tAC = (self.t2 * np.exp(1j * self._p5 * ky)
               + self.t6 * np.exp(-1j * self._p6 * ky)
               + 2 * self.t9 * np.cos(self._p1 * kx) * np.exp(-1j * self._p6 * ky))

        tAD = 4 * self.t5 * np.cos(self._p2 * kx) * np.cos(self._p8 * ky)

        # Interlayer terms
        tADp = ((4 * self.t3p * np.cos(self._p1 * kx) + 2 * self.t2p)
                * np.cos(self._q2 * ky))

        tACp = ((2 * self.t1p * np.exp(1j * self._p5 * ky)
                 + 2 * self.t4p * np.exp(-1j * self._p10 * ky))
                * np.cos(self._p1 * kx))

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
        tAA, tAB, tAC, tAD, tADp, tACp = self.basic_block(k_points, twist_angle=0.0)
        a_t = tAA + tAD + self._pf_top * tADp
        z_t = tAB + tAC + self._pf_top * tACp
        ham_top = np.zeros((num_k, 2, 2), dtype=np.complex128)
        ham_top[:, 0, 0] = a_t
        ham_top[:, 1, 1] = a_t
        ham_top[:, 0, 1] = z_t
        ham_top[:, 1, 0] = np.conj(z_t)

        if self.N_bottom == 0:
            return ham_top

        # Bottom layer (twisted)
        tAA, tAB, tAC, tAD, tADp, tACp = self.basic_block(k_points, twist_angle=self.twist_angle)
        a_b = tAA + tAD + self._pf_bot * tADp
        z_b = tAB + tAC + self._pf_bot * tACp
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
        c = np.cos(twist_angle)
        s = np.sin(twist_angle)
        kx_tmp, ky_tmp = k_points[:, 0], k_points[:, 1]
        kx = kx_tmp * c - ky_tmp * s
        ky = kx_tmp * s + ky_tmp * c

        # Aliases for precomputed geometric projections
        p1, p2, p3, p4, p5, p6 = self._p1, self._p2, self._p3, self._p4, self._p5, self._p6
        p8, p9, p10 = self._p8, self._p9, self._p10
        q1, q2 = self._q1, self._q2

        # === Local derivatives of each element ===
        # dtAA/dkx, dtAA/dky
        dtAA_x = (-2 * self.t3 * p1 * np.sin(p1 * kx)
                  - 4 * self.t10 * p1 * np.sin(p1 * kx) * np.cos(q1 * ky))
        dtAA_y = (-2 * self.t7 * q1 * np.sin(q1 * ky)
                  - 4 * self.t10 * np.cos(p1 * kx) * q1 * np.sin(q1 * ky))

        # dtAB/dkx, dtAB/dky
        e1 = np.exp(-1j * p3 * ky)
        e4 = np.exp(1j * p4 * ky)
        dtAB_x = (-2 * self.t1 * p2 * np.sin(p2 * kx) * e1
                  - 2 * self.t4 * p2 * np.sin(p2 * kx) * e4
                  - 2 * self.t8 * p9 * np.sin(p9 * kx) * e1)
        dtAB_y = (2 * self.t1 * np.cos(p2 * kx) * (-1j * p3) * e1
                  + 2 * self.t4 * np.cos(p2 * kx) * (1j * p4) * e4
                  + 2 * self.t8 * np.cos(p9 * kx) * (-1j * p3) * e1)

        # dtAC/dkx, dtAC/dky
        e5 = np.exp(1j * p5 * ky)
        e6 = np.exp(-1j * p6 * ky)
        dtAC_x = -2 * self.t9 * p1 * np.sin(p1 * kx) * e6
        dtAC_y = (self.t2 * (1j * p5) * e5
                  + self.t6 * (-1j * p6) * e6
                  + 2 * self.t9 * np.cos(p1 * kx) * (-1j * p6) * e6)

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
        c = np.cos(twist_angle)
        s = np.sin(twist_angle)
        kx_tmp, ky_tmp = k_points[:, 0], k_points[:, 1]
        kx = kx_tmp * c - ky_tmp * s
        ky = kx_tmp * s + ky_tmp * c

        # Aliases for precomputed geometric projections
        p1, p2, p3, p4, p5, p6 = self._p1, self._p2, self._p3, self._p4, self._p5, self._p6
        p8, p9, p10 = self._p8, self._p9, self._p10
        q1, q2 = self._q1, self._q2

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
        d2tAB_xx = (-2 * self.t1 * p2**2 * cos_p2kx * e1
                    - 2 * self.t4 * p2**2 * cos_p2kx * e4
                    - 2 * self.t8 * p9**2 * cos_p9kx * e1)
        d2tAB_yy = (-2 * self.t1 * p3**2 * cos_p2kx * e1
                    - 2 * self.t4 * p4**2 * cos_p2kx * e4
                    - 2 * self.t8 * p3**2 * cos_p9kx * e1)
        d2tAB_xy = (2 * self.t1 * p2 * p3 * 1j * sin_p2kx * e1
                    - 2 * self.t4 * p2 * p4 * 1j * sin_p2kx * e4
                    + 2 * self.t8 * p9 * p3 * 1j * sin_p9kx * e1)

        # tAC
        d2tAC_xx = -2 * self.t9 * p1**2 * cos_p1kx * e6
        d2tAC_yy = (-self.t2 * p5**2 * e5 - self.t6 * p6**2 * e6
                    - 2 * self.t9 * p6**2 * cos_p1kx * e6)
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

        vx_t, vy_t = self._layer_velocity(k_points, 0.0, self._pf_top)

        if self.N_bottom == 0:
            return vx_t, vy_t

        vx_b, vy_b = self._layer_velocity(k_points, self.twist_angle, self._pf_bot)

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

        wxx_t, wyy_t, wxy_t = self._layer_curvature(k_points, 0.0, self._pf_top)

        if self.N_bottom == 0:
            return wxx_t, wyy_t, wxy_t

        wxx_b, wyy_b, wxy_b = self._layer_curvature(k_points, self.twist_angle, self._pf_bot)

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
# Single-particle part
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


def plot_layer_projected_unfolded_bands(N_top=4, N_bottom=4, twist_angle=0.0,
                                        k_fine_steps=360, y_lim=(-2, 1.5),
                                        lw=2.5, save_prefix=""):
    """
    Plot unfolded bands colored by layer polarization.

    The layer polarization is computed from the eigenvector weight in the
    sublattice basis:

        P_layer = W_top - W_bottom

    with W_top = sum(|u_0|^2 + |u_1|^2) and
    W_bottom = sum(|u_2|^2 + |u_3|^2). P_layer=+1 is top-layer polarized
    and is plotted red; P_layer=-1 is bottom-layer polarized and is plotted
    blue. Mixed states appear in the middle of the colormap.
    """
    if N_bottom == 0:
        raise ValueError("Layer-projected bands require N_bottom > 0.")

    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom,
                           twist_angle=twist_angle)

    b_lat = model.b_lat
    a_lat = model.a_lat
    dg = 2 * np.pi * np.abs(1 / b_lat - 1 / a_lat)

    n_seg = k_fine_steps // 2
    path_1 = np.zeros((n_seg, 2))
    path_1[:, 0] = np.linspace(-dg / 2, 0, n_seg, endpoint=False)
    path_2 = np.zeros((n_seg + 1, 2))
    path_2[:, 1] = np.linspace(0, dg / 2, n_seg + 1)
    k_path = np.vstack([path_1, path_2])

    dists = np.linalg.norm(np.diff(k_path, axis=0), axis=1)
    k_dist = np.concatenate([[0], np.cumsum(dists)])
    sym_pos = [0.0, k_dist[n_seg], k_dist[-1]]
    sym_labels = [r'$X$', r'$\Gamma$', r'$Y$']

    print(f"Calculating layer-projected unfolded bands ({len(k_path)} k-points)...")
    H = model.get_hamiltonians(k_path)
    evals, evecs = np.linalg.eigh(H)
    nbnd = evals.shape[1]
    vbm = np.max(evals[:, :nbnd // 2])
    energies = evals - vbm

    top_weight = np.sum(np.abs(evecs[:, 0:2, :])**2, axis=1)
    bottom_weight = np.sum(np.abs(evecs[:, 2:4, :])**2, axis=1)
    norm = top_weight + bottom_weight
    layer_pol = np.zeros_like(top_weight)
    valid = norm > 1e-14
    layer_pol[valid] = (top_weight[valid] - bottom_weight[valid]) / norm[valid]

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    cmap = plt.get_cmap('bwr')
    color_norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

    for ib in range(nbnd):
        points = np.column_stack([k_dist, energies[:, ib]])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        seg_pol = 0.5 * (layer_pol[:-1, ib] + layer_pol[1:, ib])
        lc = LineCollection(segments, cmap=cmap, norm=color_norm,
                            linewidths=lw, alpha=0.9)
        lc.set_array(seg_pol)
        ax.add_collection(lc)

    for pos in sym_pos:
        ax.axvline(pos, c='gray', ls='-', lw=0.5)
    ax.axhline(0, c='k', ls='--', lw=0.5, alpha=0.5, zorder=0)
    ax.set_xticks(sym_pos)
    ax.set_xticklabels(sym_labels, fontsize=12)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(y_lim)
    ax.set_ylabel("Energy - VBM (eV)", fontsize=12)
    # ax.set_title(f'Layer-Projected Unfolded Bands\n'
    #              f'N={N_top}/{N_bottom}, twist={np.degrees(twist_angle):.0f} deg')
    plt.tick_params(direction='in', labelsize=10)
    ax.grid(True, alpha=0.3)

    sm = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    # cbar.set_label(r'Layer Polarization $P_\mathrm{layer}$', rotation=270, labelpad=15)
    cbar.set_ticks([-1,1])
    cbar.set_ticklabels(['Bottom','Top'],fontsize=12)

    plt.tight_layout()
    suffix = f"N{N_top}_{N_bottom}_tw{np.degrees(twist_angle):.0f}{save_prefix}"
    fname = f"EM_layer_projected_unfolded_{suffix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved Layer-Projected Unfolded Bands: {fname}")

    rows = []
    for ik in range(len(k_path)):
        for ib in range(nbnd):
            rows.append([
                ik, ib, k_dist[ik], k_path[ik, 0], k_path[ik, 1],
                evals[ik, ib], energies[ik, ib],
                top_weight[ik, ib], bottom_weight[ik, ib], layer_pol[ik, ib],
            ])
    _save_dat(
        f"EM_layer_projected_unfolded_{suffix}.dat",
        np.array(rows, dtype=float),
        "k_index band_index k_dist kx ky energy_eV energy_minus_vbm_eV "
        "top_weight bottom_weight layer_polarization",
        fmt=["%d", "%d", "%.10e", "%.10e", "%.10e",
             "%.10e", "%.10e", "%.10e", "%.10e", "%.10e"]
    )

    return k_dist, evals, top_weight, bottom_weight, layer_pol


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
    unfolded_cols = [k_dist] + [unfolded_E[:, ib] - vbm for ib in range(nbnd)]
    _save_dat(
        f"EM_unfolded_{suffix}.dat",
        np.column_stack(unfolded_cols),
        "k_dist " + " ".join([f"E{ib}_minus_vbm_eV" for ib in range(nbnd)])
    )

    # Folded plot
    if len(folded_k) > 0:
        plt.figure(figsize=(4.8, 4.8))
        folded_sym_pos = [0.0, k_boundary, 2*k_boundary]
        folded_sym_labels = [r"$X'$", r'$\Gamma$', r"$Y'$"]
        if folded_is_structured:
            # Structured: folded_k is (n_k,), folded_E is (n_k, n_bands_folded)
            plt.plot(folded_k, folded_E[:, 0] - vbm, 'r-', lw=2.5, alpha=0.5, label='Folded Bands', zorder=1)
            plt.plot(folded_k, folded_E[:, 1:] - vbm, 'r-', lw=2.5, alpha=0.5)
        else:
            # Legacy scatter format
            plt.scatter(folded_k, folded_E - vbm, s=20, color='red', alpha=0.9,
                        label='Folded Bands', facecolors='white', edgecolors='red',
                        linewidths=0.5, zorder=0)
        for pos in folded_sym_pos:
            plt.axvline(pos, c='gray', ls='--', lw=0.5,zorder=0)

        # # # Add the DFT data
        # # vasp_dat = np.loadtxt("571.dat")
        # # vasp_kpath = vasp_dat[:, 0]
        # # vasp_kpath = vasp_kpath / max(vasp_kpath) * max(folded_k) # normalize to our k_dist
        # # vasp_energies = vasp_dat[:, 1] + 0.271584 # set VBM to zero
        # # vasp_energies = vasp_energies.reshape(848,40)
        # # vasp_energies[700:,:] += 1.015

        # # plt.scatter(vasp_kpath, vasp_energies, s=36, color='black', alpha=0.8, label='VASP Bands',
        # #             facecolors='white', edgecolors='black', linewidths=1.6, zorder=0)
            
        # Add the DFT data
        vasp_dat = np.loadtxt("571-33.dat")
        vasp_kpath = vasp_dat[:, 0]
        vasp_kpath = vasp_kpath / max(vasp_kpath) * max(folded_k) # normalize to our k_dist
        vasp_energies = vasp_dat[:, 1] + 0.0554 # set VBM to zero
        vasp_energies = vasp_energies.reshape(2528,30)
        vasp_energies[2100:,:] += 0.560

        plt.scatter(vasp_kpath, vasp_energies, s=32, color='black', alpha=0.8, label='VASP Bands',
                    facecolors='white', edgecolors='black', linewidths=1.2, zorder=0)


        plt.xticks(folded_sym_pos, folded_sym_labels, fontsize=10)
        plt.tick_params(direction='in', labelsize=10)
        plt.ylim(y_lim)
        plt.xlim(0, 2*k_boundary)
        plt.ylabel("Energy (eV)",fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"EM_folded_{suffix}.png", dpi=200)
        plt.close()
        if folded_is_structured:
            folded_cols = [folded_k] + [folded_E[:, ib] - vbm for ib in range(folded_E.shape[1])]
            folded_header = "k_dist " + " ".join(
                [f"E{ib}_minus_vbm_eV" for ib in range(folded_E.shape[1])]
            )
            folded_data = np.column_stack(folded_cols)
        else:
            folded_data = np.column_stack([folded_k, folded_E - vbm])
            folded_header = "k_dist E_minus_vbm_eV"
        _save_dat(f"EM_folded_{suffix}.dat", folded_data, folded_header)

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
    band_indices = list(range(start_band, end_band))
    band_cols = [evals[:, ib] for ib in band_indices]
    _save_dat(
        f"EM_3D{save_prefix}.dat",
        np.column_stack([k_points[:, 0], k_points[:, 1]] + band_cols),
        "kx ky " + " ".join([f"E_band{ib}_eV" for ib in band_indices])
    )


def calculate_ipa_dielectric_function(N_top=1, N_bottom=1, twist_angle=0.0,
                                      E_range=(0.0, 1.0), n_E=500, eta=0.010,
                                      k_range=0.15, n_k=60, layerthickness=5.2,
                                      eps_inf=1.0):
    r"""
    Calculate IPA dielectric function from interband transitions.

        \epsilon_2^{ii}(\omega) \propto
        \sum_{v,c} \int_{BZ} d^2k
        |\langle c,k | v_i | v,k \rangle|^2 / (E_{c,k} - E_{v,k})^2
        \cdot \delta(E_{c,k} - E_{v,k} - \hbar \omega)
    """
    print(f"Calculating IPA dielectric function...")
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

    omegas = np.linspace(E_range[0], E_range[1]*2, n_E*2)
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

    e2_coulomb = 14.3996454784255  # e^2/(4*pi*eps0), eV*Angstrom
    a_lat = model.a_lat
    b_lat = model.b_lat
    A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2
    V_uc = A_uc * layerthickness * (N_top + N_bottom)
    pref = 16 * np.pi * e2_coulomb / V_uc

    eps2_xx = pref * sigma_xx
    eps2_yy = pref * sigma_yy
    eps1_xx = kramers_kronig_eps1_from_eps2(omegas, eps2_xx, eps_inf=eps_inf)
    eps1_yy = kramers_kronig_eps1_from_eps2(omegas, eps2_yy, eps_inf=eps_inf)

    eps_data = {
        'eps1_xx': eps1_xx,
        'eps2_xx': eps2_xx,
        'eps1_yy': eps1_yy,
        'eps2_yy': eps2_yy,
        'sigma_xx_raw': sigma_xx,
        'sigma_yy_raw': sigma_yy,
    }
    return omegas, model, eps_data


def calculate_optical_conductivity(N_top=1, N_bottom=1, twist_angle=0.0,
                                   E_range=(0.0, 1.0), n_E=500, eta=0.010,
                                   k_range=0.15, n_k=60, layerthickness=5.2,
                                   save_prefix=""):
    r"""
    Calculate IPA dielectric function and omega * epsilon_2 spectrum.
    """
    omegas, _, eps_data = calculate_ipa_dielectric_function(
        N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle,
        E_range=E_range, n_E=n_E, eta=eta,
        k_range=k_range, n_k=n_k, layerthickness=layerthickness,
    )
    eps1_xx = eps_data['eps1_xx']
    eps2_xx = eps_data['eps2_xx']
    eps1_yy = eps_data['eps1_yy']
    eps2_yy = eps_data['eps2_yy']
    sigma_xx = eps_data['sigma_xx_raw']
    sigma_yy = eps_data['sigma_yy_raw']
    absorption_xx = omegas * eps2_xx
    absorption_yy = omegas * eps2_yy

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
    _save_dat(
        f"EM_absorption{save_prefix}.dat",
        np.column_stack([
            omegas,
            eps1_xx, eps2_xx,
            eps1_yy, eps2_yy,
            absorption_xx, absorption_yy,
            sigma_xx, sigma_yy,
        ]),
        "omega_eV eps1_xx eps2_xx eps1_yy eps2_yy "
        "omega_eps2_xx omega_eps2_yy sigma_xx_raw sigma_yy_raw"
    )


def calculate_current(N_top=1, N_bottom=1, twist_angle=0.0, layerthickness=5.2,
                      E_range=(0.0, 1.0), n_E=500, eta=0.010,
                      k_range=0.15, n_k=60, band_window=None,n_k_bse=30,
                      intensity_W_cm2=1.6e4, sample_width_um=2.0,
                      shift_source="ipa", shift_results=None,
                      n_val=2, n_cond=2,
                      kappa=2.5, r0=5.0,
                      save_prefix="", **bse_kwargs):
    r"""
    Compute photocurrent from z-shift-current conductivity.

        I(omega) = (1 - R) * sigma_zbb(omega) * 2 * I_light * d * w
                   / (epsilon0 * c)

    The z-shift conductivity is expected in microampere/V^2, matching
    calculate_z_shift_current and calculate_bse_z_shift_current outputs.

    Parameters
    ----------
    shift_source : {"ipa", "bse", "provided"}
        "ipa" calls calculate_z_shift_current, "bse" calls
        calculate_bse_z_shift_current, and "provided" uses shift_results.
    shift_results : tuple or dict
        For "provided", pass either (omegas, results) or a dict with
        keys "omegas" and "results".
    """
    total_thickness_A = layerthickness * (N_top + N_bottom)

    print("\n[1] Computing IPA dielectric function for reflectivity...")
    omegas_eps, _, eps_data = calculate_ipa_dielectric_function(
        N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle,
        E_range=E_range, n_E=n_E, eta=eta,
        k_range=k_range, n_k=n_k,
        layerthickness=layerthickness,
    )
    eps1_xx, eps2_xx = eps_data['eps1_xx'], eps_data['eps2_xx']
    eps1_yy, eps2_yy = eps_data['eps1_yy'], eps_data['eps2_yy']
    refl_xx = normal_incidence_reflectivity(eps1_xx, eps2_xx)
    refl_yy = normal_incidence_reflectivity(eps1_yy, eps2_yy)

    print(f"\n[2] Getting z-shift-current conductivity ({shift_source})...")
    if shift_source == "ipa":
        omegas_sc, sc_results = calculate_z_shift_current(
            N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle,
            E_range=E_range, n_E=n_E, eta=eta,
            k_range=k_range, n_k=n_k,
            layerthickness=layerthickness,
            band_window=band_window,
            save_prefix=save_prefix,
        )
    elif shift_source == "bse":
        omegas_sc, sc_results, _, _ = calculate_bse_z_shift_current(
            N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle,
            E_range=E_range, n_E=n_E, eta=eta,
            k_range=k_range,n_k_bse =n_k_bse,
            thickness=layerthickness,
            band_window=band_window,
            save_prefix=save_prefix,
            n_val=n_val, n_cond=n_cond,
            kappa=kappa, r0=r0,
            **bse_kwargs,
        )
    elif shift_source == "provided":
        if shift_results is None:
            raise ValueError("shift_results must be provided when shift_source='provided'.")
        if isinstance(shift_results, dict):
            omegas_sc = shift_results['omegas']
            sc_results = shift_results['results']
        else:
            omegas_sc, sc_results = shift_results[:2]
    else:
        raise ValueError("shift_source must be 'ipa', 'bse', or 'provided'.")

    if len(omegas_sc) != len(omegas_eps) or not np.allclose(omegas_sc, omegas_eps):
        refl_xx = np.interp(omegas_sc, omegas_eps, refl_xx)
        refl_yy = np.interp(omegas_sc, omegas_eps, refl_yy)
        eps1_xx = np.interp(omegas_sc, omegas_eps, eps1_xx)
        eps2_xx = np.interp(omegas_sc, omegas_eps, eps2_xx)
        eps1_yy = np.interp(omegas_sc, omegas_eps, eps1_yy)
        eps2_yy = np.interp(omegas_sc, omegas_eps, eps2_yy)

    sigma_zxx = sc_results[('z', 'x', 'x')]
    sigma_zyy = sc_results[('z', 'y', 'y')]
    current_zxx_A = shift_conductivity_to_current(
        sigma_zxx, refl_xx, total_thickness_A, sample_width_um,
        intensity_W_cm2=intensity_W_cm2,
    )
    current_zyy_A = shift_conductivity_to_current(
        sigma_zyy, refl_yy, total_thickness_A, sample_width_um,
        intensity_W_cm2=intensity_W_cm2,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(omegas_sc, current_zxx_A * 1.0e9, 'r-', lw=2, label=r'$I_{zxx}$')
    plt.plot(omegas_sc, current_zyy_A * 1.0e9, 'b--', lw=2, label=r'$I_{zyy}$')
    plt.axhline(0, color='k', lw=0.5, ls='--')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Current (nA)')
    plt.title(rf'$I={intensity_W_cm2:.2e}$ W/cm$^2$, $w={sample_width_um:g}$ $\mu$m')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)
    fname = f"EM_current_{shift_source}{save_prefix}_{N_top}_{N_bottom}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved current figure: {fname}")

    _save_dat(
        f"EM_current_{shift_source}{save_prefix}_{N_top}_{N_bottom}.dat",
        np.column_stack([
            omegas_sc,
            current_zxx_A, current_zyy_A,
            sigma_zxx, sigma_zyy,
            refl_xx, refl_yy,
            eps1_xx, eps2_xx, eps1_yy, eps2_yy,
        ]),
        "omega_eV current_zxx_A current_zyy_A "
        "sigma_zxx_uA_per_V2 sigma_zyy_uA_per_V2 "
        "reflectivity_x reflectivity_y eps1_xx eps2_xx eps1_yy eps2_yy"
    )
    return omegas_sc, {
        'current_zxx_A': current_zxx_A,
        'current_zyy_A': current_zyy_A,
        'reflectivity_x': refl_xx,
        'reflectivity_y': refl_yy,
        'sigma_zxx_uA_per_V2': sigma_zxx,
        'sigma_zyy_uA_per_V2': sigma_zyy,
        'eps1_xx': eps1_xx,
        'eps2_xx': eps2_xx,
        'eps1_yy': eps1_yy,
        'eps2_yy': eps2_yy,
    }


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
    _save_dat(
        f"EM_M_B{band_i}-{band_j}{save_prefix}.dat",
        np.column_stack([k_points[:, 0], k_points[:, 1], np.abs(M_x)**2, np.abs(M_y)**2]),
        "kx ky Mx_abs2 My_abs2"
    )


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
    z_op[0, 0] = +layerthickness * N_top / 2.0  # top basis 0
    z_op[1, 1] = +layerthickness * N_top / 2.0  # top basis 1
    if dim_H == 4:
        z_op[2, 2] = -layerthickness * N_bottom / 2.0  # bottom basis 0
        z_op[3, 3] = -layerthickness * N_bottom / 2.0  # bottom basis 1

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
    _save_dat(
        f"EM_z_sc{save_prefix}.dat",
        np.column_stack([
            omegas,
            results[('z', 'x', 'x')],
            results[('z', 'y', 'y')],
        ]),
        "omega_eV sigma_zxx_uA_per_V2 sigma_zyy_uA_per_V2"
    )
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
    _save_dat(
        f"EM_sc{save_prefix}.dat",
        np.column_stack([
            omegas,
            results[('x', 'x', 'x')],
            results[('x', 'y', 'y')],
            results[('y', 'x', 'x')],
            results[('y', 'y', 'y')],
        ]),
        "omega_eV sigma_xxx_uA_A_per_V2 sigma_xyy_uA_A_per_V2 "
        "sigma_yxx_uA_A_per_V2 sigma_yyy_uA_A_per_V2"
    )
    return omegas, results


def plot_bandgap_scaling(N_top=1, N_bottom=1, twist_angle=0.0,
                         E_g=2.1, gamma_c = 0.58, gamma_v = -0.32,):
    """
    Plot how the gap changes with number of bottom layer changing.
    """
    # Get the bandgap of my model
    gap_level = []
    sub_gap_level = []
    for N_bot in range(N_bottom[0], N_bottom[1]):
        model = TwistedBPModel(N_top=N_top, N_bottom=N_bot, twist_angle=twist_angle)
        k_points = np.array([[0.0, 0.0]])
        evals, _ = np.linalg.eigh(model.get_hamiltonians(k_points))
        mid = evals.shape[1] // 2
        E_gap = evals[0, mid] - evals[0, mid-1]
        E_subgap = evals[0, mid] - evals[0, mid-2]
        gap_level.append(E_gap)
        sub_gap_level.append(E_subgap)
        print(f"N_bottom={N_bot}: Bandgap = {E_gap:.3f} eV")

    # Get the analytic levels, ref: Huang et al., Science 386, 526–531 (2024)
    level_list = np.linspace(N_bottom[0], N_bottom[1]+3, 500)
    Y_bright = E_g - 2 * gamma_c * np.cos(np.pi/(level_list + N_top + 1)) + 2 * gamma_v * np.cos(np.pi/(N_top + 1))
    X_bright = E_g - 2 * gamma_c * np.cos(np.pi/(level_list + N_top + 1)) + 2 * gamma_v * np.cos(np.pi/(level_list + 1))
    gap_level = np.array(gap_level)
    sub_gap_level = np.array(sub_gap_level)

    # Plotting
    plt.figure(figsize=(5, 5))
    plt.plot(level_list, X_bright, label='X-bright (analytic)', color='red', ls='--')
    plt.plot(level_list, Y_bright, label='Y-bright (analytic)', color='blue', ls='--')
    plt.scatter(range(N_bottom[0], N_bottom[1]), gap_level, label='model', color='red', marker='o')
    plt.scatter(range(N_bottom[0], N_bottom[1]), sub_gap_level, label='sub-gap (model)', color='blue', marker='o')
    # plt.tight_layout()
    plt.ylim(0.2,1.6)
    plt.legend()
    plt.xlabel('N_bottom')
    plt.ylabel('Bandgap (eV)')
    fname = f"gap.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved: {fname}")
    _save_dat(
        "gap_model.dat",
        np.column_stack([np.arange(N_bottom[0], N_bottom[1]), gap_level]),
        "N_bottom bandgap_eV"
    )
    _save_dat(
        "gap_analytic.dat",
        np.column_stack([level_list, X_bright, Y_bright]),
        "N_bottom_continuous X_bright_eV Y_bright_eV"
    )


def calculate_effective_mass(N_top=1, N_bottom=1, twist_angle=0.0, k_max=0.01, n_points=51):
    """
    Calculate effective masses by fitting a parabola E(k) = a k^2 + b k + c
    in the range [-k_max, k_max] around the Gamma point.

    Note: [Electron] m_x*:  0.2313 m_e, m_y*:  0.2313 m_e
          [Hole]     m_x*:  0.6512 m_e, m_y*:  0.1400 m_e
    """
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)
    hbar2_over_me = 7.619964 # eV·Å^2

    k_vals = np.linspace(-k_max, k_max, n_points)
    
    # x-direction: (k, 0)
    k_points_x = np.column_stack([k_vals, np.zeros_like(k_vals)])
    H_x = model.get_hamiltonians(k_points_x)
    evals_x, _ = np.linalg.eigh(H_x)
    
    # y-direction: (0, k)
    k_points_y = np.column_stack([np.zeros_like(k_vals), k_vals])
    H_y = model.get_hamiltonians(k_points_y)
    evals_y, _ = np.linalg.eigh(H_y)
    
    Nb = evals_x.shape[1]
    idx_v = Nb // 2 - 2  # VBM
    idx_c = Nb // 2      # CBM
    
    E_v_x = evals_x[:, idx_v]
    E_c_x = evals_x[:, idx_c]
    E_v_y = evals_y[:, idx_v]
    E_c_y = evals_y[:, idx_c]
    
    # Fit parabola: a*x^2 + b*x + c
    # The second derivative is 2*a
    p_c_x = np.polyfit(k_vals, E_c_x, 2)
    p_c_y = np.polyfit(k_vals, E_c_y, 2)
    
    # For valence band, curvature is negative, so we fit -E_v setting hole mass positive
    p_v_x = np.polyfit(k_vals, -E_v_x, 2)
    p_v_y = np.polyfit(k_vals, -E_v_y, 2)
    
    # m* = hbar^2 / (2a)
    m_e_x = hbar2_over_me / (2 * p_c_x[0])
    m_e_y = hbar2_over_me / (2 * p_c_y[0])
    m_h_x = hbar2_over_me / (2 * p_v_x[0])
    m_h_y = hbar2_over_me / (2 * p_v_y[0])
    
    print("="*60)
    print(f"Effective Mass (N_top={N_top}, N_bottom={N_bottom}, twist={np.degrees(twist_angle):.1f}°)")
    print(f"Fitting range: [-{k_max}, {k_max}] Å^-1 with {n_points} points.")
    print("="*60)
    print(f"[Electron] m_x*: {m_e_x:7.4f} m_e, m_y*: {m_e_y:7.4f} m_e")
    print(f"[Hole]     m_x*: {m_h_x:7.4f} m_e, m_y*: {m_h_y:7.4f} m_e")
    print("="*60)

    return (m_e_x, m_e_y), (m_h_x, m_h_y)


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


def _torch_cuda_is_available():
    return torch is not None and torch.cuda.is_available()


def _resolve_bse_torch_settings(use_gpu="auto", gpu_dtype="complex64"):
    """
    Pick the BSE backend. 3060-class cards are much faster in complex64 than
    complex128, while the CPU fallback keeps complex128 for numerical parity.
    """
    if use_gpu == "auto":
        enabled = _torch_cuda_is_available()
    else:
        enabled = bool(use_gpu)

    if not enabled:
        return False, None, None, None
    if torch is None:
        print("  GPU requested but PyTorch is not installed; falling back to NumPy/SciPy CPU.")
        return False, None, None, None
    if not torch.cuda.is_available():
        print("  GPU requested but CUDA is not available in this environment; falling back to NumPy/SciPy CPU.")
        return False, None, None, None

    dtype_name = str(gpu_dtype).lower()
    if dtype_name in ("complex64", "c64", "float32"):
        complex_dtype = torch.complex64
        real_dtype = torch.float32
    elif dtype_name in ("complex128", "c128", "float64"):
        complex_dtype = torch.complex128
        real_dtype = torch.float64
    else:
        raise ValueError("gpu_dtype must be 'complex64' or 'complex128'")

    return True, torch.device("cuda"), complex_dtype, real_dtype


def _keldysh_potential_torch(q, kappa=2.5, r0=5.0, N_top=1, N_bottom=1):
    V = torch.zeros_like(q)
    mask = q > 1e-12
    V[mask] = 14.3996 * 2 * np.pi / (
        kappa * (q[mask] * (1 + r0 * (N_top + N_bottom) * q[mask]))
    )
    return V


def build_bse_hamiltonian_gpu(evals, evecs, k_points, v_idx, c_idx, A_uc,
                              kappa=2.5, r0=5.0, N_top=1, N_bottom=1,
                              device=None, complex_dtype=None, real_dtype=None):
    """
    CUDA implementation of the dense BSE matrix build.

    The basis order is identical to the NumPy path: flat(k, v, c), so downstream
    spectra and exciton analysis remain unchanged.
    """
    if device is None:
        device = torch.device("cuda")
    if complex_dtype is None:
        complex_dtype = torch.complex64
    if real_dtype is None:
        real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64

    Nk = len(k_points)
    Nv = len(v_idx)
    Nc = len(c_idx)
    dim_bse = Nv * Nc * Nk

    print(f"  Building BSE Hamiltonian on GPU: {Nv}v x {Nc}c x {Nk}k = {dim_bse} basis states")
    print(f"    Dense matrix memory: {dim_bse**2 * torch.empty((), dtype=complex_dtype).element_size() / 1e9:.2f} GB")

    evals_t = torch.as_tensor(evals, dtype=real_dtype, device=device)
    evecs_t = torch.as_tensor(evecs, dtype=complex_dtype, device=device)
    k_points_t = torch.as_tensor(k_points, dtype=real_dtype, device=device)
    v_idx_t = torch.as_tensor(v_idx, dtype=torch.long, device=device)
    c_idx_t = torch.as_tensor(c_idx, dtype=torch.long, device=device)

    E_v = evals_t.index_select(1, v_idx_t)
    E_c = evals_t.index_select(1, c_idx_t)
    delta_E = E_c[:, None, :] - E_v[:, :, None]
    diag_vals = delta_E.reshape(dim_bse)

    H_bse = torch.diag(diag_vals).to(complex_dtype)

    dk = k_points_t[:, None, :] - k_points_t[None, :, :]
    q_mag = torch.linalg.vector_norm(dk, dim=2)
    Vq = _keldysh_potential_torch(q_mag, kappa=kappa, r0=r0,
                                  N_top=N_top, N_bottom=N_bottom) / (Nk * A_uc)

    U_c = evecs_t.index_select(2, c_idx_t)
    U_v = evecs_t.index_select(2, v_idx_t)

    print(f"    Computing overlaps on GPU...")
    overlap_cc = torch.einsum('kai,laj->klij', U_c.conj(), U_c)
    overlap_vv = torch.einsum('lai,kaj->lkij', U_v.conj(), U_v)

    print(f"    Assembling dense kernel on GPU...")
    kernel = torch.einsum('kl,klij,lkpv->kvilpj', Vq, overlap_cc, overlap_vv)
    H_bse -= kernel.reshape(dim_bse, dim_bse)
    del kernel, overlap_cc, overlap_vv, Vq, dk, q_mag, U_c, U_v

    H_bse = 0.5 * (H_bse + H_bse.conj().T)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        used = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"    CUDA peak allocated so far: {used:.2f} GB")

    return H_bse


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

    print(f"    Assembling dense kernel (single vectorized einsum)...")
    kernel = np.einsum('kl,klij,lkpv->kvilpj', Vq, overlap_cc, overlap_vv, optimize=True)
    H_bse -= kernel.reshape(dim_bse, dim_bse)
    del kernel, overlap_cc, overlap_vv, Vq, dk, q_mag

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


def _run_bse_pipeline(model, k_range, n_k_bse, n_val, n_cond,
                      thickness=None, kappa=2.5, r0=5.0, band_window=None,
                      use_gpu="auto", gpu_dtype="complex64",
                      gpu_full_eigh_max_dim=32000):
    """
    Single-particle solve + BSE diagonalization + degenerate-subspace resolution.

    Encapsulates the shared boilerplate used by all BSE analysis functions.
    Returns a dict with all intermediate quantities; callers extract what they need.

    When *thickness* is not None, also computes the z-operator diagonal elements.
    """
    # 1. k-grid
    kx = np.linspace(-k_range, k_range, n_k_bse)
    ky = np.linspace(-k_range, k_range, n_k_bse)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)
    dk = kx[1] - kx[0]

    # 2. Single-particle solve
    print(f"  Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)
    Nb = evals.shape[1]

    # 3. Active band selection
    mid = Nb // 2
    if band_window is not None:
        v_idx = np.arange(band_window[0], band_window[1] + 1)
        c_idx = np.arange(band_window[2], band_window[3] + 1)
    else:
        v_idx = np.arange(mid - n_val, mid)
        c_idx = np.arange(mid, mid + n_cond)
    Nv, Nc = len(v_idx), len(c_idx)
    qp_gap = np.min(evals[:, c_idx[0]] - evals[:, v_idx[-1]])
    print(f"  Bands: valence {v_idx}, conduction {c_idx}, QP gap: {qp_gap:.4f} eV")

    # 4. Velocity in eigenbasis
    print(f"  Computing velocity matrices...")
    vx_orb, vy_orb = model.get_velocity_matrices(k_points)
    U = evecs
    U_dag = np.conj(np.transpose(U, (0, 2, 1)))
    vx_eig = U_dag @ vx_orb @ U
    vy_eig = U_dag @ vy_orb @ U

    # 4b. Generalized derivative (curvature) matrices in eigenbasis
    w_xx_orb, w_yy_orb, w_xy_orb = model.get_generalized_derivative_matrices(k_points)
    w_xx_eig = U_dag @ w_xx_orb @ U
    w_yy_eig = U_dag @ w_yy_orb @ U
    w_xy_eig = U_dag @ w_xy_orb @ U

    # 5. Interband position matrix elements  r^b_{cv} = v^b_{cv} / (i * omega_{cv})
    E_v = evals[:, v_idx]
    E_c = evals[:, c_idx]
    dE = E_c[:, None, :] - E_v[:, :, None]  # (Nk, Nv, Nc)

    eps_denom = 1e-5
    r_b = {}
    for b_dir, v_b in [('x', vx_eig), ('y', vy_eig)]:
        vb_cv = np.transpose(v_b[:, c_idx, :][:, :, v_idx], (0, 2, 1))
        rb = np.zeros_like(vb_cv)
        valid = np.abs(dE) > eps_denom
        rb[valid] = vb_cv[valid] / (1j * dE[valid])
        r_b[b_dir] = rb

    # 6. z-operator (only when thickness is provided)
    z_eig = None
    z_diag = None
    delta_z = None
    if thickness is not None:
        z_op = np.zeros((Nb, Nb), dtype=np.float64)
        z_op[0, 0] = +thickness * model.N_top / 2.0
        z_op[1, 1] = +thickness * model.N_top / 2.0
        if Nb == 4:
            z_op[2, 2] = -thickness * model.N_bottom / 2.0
            z_op[3, 3] = -thickness * model.N_bottom / 2.0
        z_eig = U_dag @ z_op @ U
        z_diag = np.real(np.diagonal(z_eig, axis1=1, axis2=2))
        z_v = z_diag[:, v_idx]
        z_c = z_diag[:, c_idx]
        delta_z = z_v[:, :, None] - z_c[:, None, :]  # (Nk, Nv, Nc)
        delta_z = delta_z * (-1)  # (f_nm = -1 for c->v)

    # 7. BSE Hamiltonian
    a_lat = model.a_lat
    b_lat = model.b_lat
    A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2
    print(f"  Moire unit cell area: {A_uc:.1f} A^2")

    dim_bse = Nv * Nc * Nk
    print(f"  Building BSE Hamiltonian ({Nv}v x {Nc}c x {Nk}k = {dim_bse} basis)...")
    gpu_enabled, device, complex_dtype, real_dtype = _resolve_bse_torch_settings(use_gpu, gpu_dtype)
    if gpu_enabled:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            props = torch.cuda.get_device_properties(device)
            print(f"  BSE backend: CUDA ({props.name}, {props.total_memory / 1e9:.1f} GB), dtype={complex_dtype}")
        H_bse = build_bse_hamiltonian_gpu(
            evals, evecs, k_points, v_idx, c_idx, A_uc,
            kappa=kappa, r0=r0, N_top=model.N_top, N_bottom=model.N_bottom,
            device=device, complex_dtype=complex_dtype, real_dtype=real_dtype,
        )
    else:
        print("  BSE backend: NumPy/SciPy CPU")
        H_bse = build_bse_hamiltonian(evals, evecs, k_points, v_idx, c_idx, A_uc,
                                      kappa=kappa, r0=r0,
                                      N_top=model.N_top, N_bottom=model.N_bottom)

    # 8. Diagonalize BSE
    dim_bse_mat = H_bse.shape[0]
    # n_exciton_max = min(1000, dim_bse_mat - 2)
    n_exciton_max = dim_bse_mat
    print(f"  Diagonalizing BSE ({dim_bse_mat}x{dim_bse_mat})...")
    t0 = time.time()
    if gpu_enabled and dim_bse_mat <= gpu_full_eigh_max_dim:
        print(f"    Using torch.linalg.eigh on CUDA (full dense spectrum)...")
        Omega_t, A_t = torch.linalg.eigh(H_bse)
        Omega_S = Omega_t.detach().cpu().numpy().astype(np.float64, copy=False)
        A_coeff = A_t.detach().cpu().numpy().astype(np.complex128, copy=False)
        del Omega_t, A_t, H_bse
        torch.cuda.empty_cache()
    elif gpu_enabled:
        print(f"    Matrix is larger than gpu_full_eigh_max_dim={gpu_full_eigh_max_dim}; "
              f"using CPU eigsh for lowest {n_exciton_max} states.")
        H_bse = H_bse.detach().cpu().numpy().astype(np.complex128, copy=False)
        torch.cuda.empty_cache()
        Omega_S, A_coeff = eigsh(H_bse, k=n_exciton_max, which='SM')
        sort_idx = np.argsort(Omega_S)
        Omega_S = Omega_S[sort_idx]
        A_coeff = A_coeff[:, sort_idx]
        del H_bse
    elif dim_bse_mat > 10000:
        print(f"    Using sparse eigsh (lowest {n_exciton_max} states)...")
        Omega_S, A_coeff = eigsh(H_bse, k=n_exciton_max, which='SM')
        sort_idx = np.argsort(Omega_S)
        Omega_S = Omega_S[sort_idx]
        A_coeff = A_coeff[:, sort_idx]
        del H_bse
    else:
        Omega_S, A_coeff = scipy_eigh(H_bse, driver='evd')
        del H_bse
    dt = time.time() - t0
    print(f"    Done in {dt:.1f} s")
    print(f"    Exciton energy range: {Omega_S[0]:.4f} - {Omega_S[-1]:.4f} eV")
    print(f"    Lowest exciton: {Omega_S[0]:.4f} eV  (QP gap ~ {np.min(dE):.4f} eV)")
    print(f"    Binding energy: {np.min(dE) - Omega_S[0]:.4f} eV")

    # 9. Resolve degenerate excitons into polarization eigenstates
    r_x_flat = r_b['x'].reshape(dim_bse)
    r_y_flat = r_b['y'].reshape(dim_bse)
    A_coeff = _resolve_degenerate_excitons(Omega_S, A_coeff, r_x_flat, r_y_flat)

    return {
        'k_points': k_points, 'KX': KX, 'KY': KY, 'Nk': Nk, 'dk': dk,
        'evals': evals, 'evecs': evecs, 'Nb': Nb,
        'v_idx': v_idx, 'c_idx': c_idx, 'Nv': Nv, 'Nc': Nc,
        'dE': dE, 'r_b': r_b, 'r_x_flat': r_x_flat, 'r_y_flat': r_y_flat,
        'z_eig': z_eig, 'z_diag': z_diag, 'delta_z': delta_z,
        'A_uc': A_uc, 'Omega_S': Omega_S, 'A_coeff': A_coeff, 'dim_bse': dim_bse,
        'U': U, 'U_dag': U_dag, 'vx_eig': vx_eig, 'vy_eig': vy_eig,
        'w_xx_eig': w_xx_eig, 'w_yy_eig': w_yy_eig, 'w_xy_eig': w_xy_eig,
    }


def _many_body_z_shift_vector(A_coeff, z_eig, v_idx, c_idx, Nk, Nv, Nc):
    """
    R^z_S0 from the full active-space z matrix.

    This implements the band-space part of Eq. (7) in ref.pdf for alpha=z:

        sum_{v,c,c',k} A*_{v c' k} A_{v c k} z^c_{c'c}(k)
      - sum_{v,v',c,k} A*_{v' c k} A_{v c k} z^v_{v v'}(k)

    The exciton-envelope derivative term is still omitted, which is the intended
    approximation for this non-periodic out-of-plane coordinate model.
    """
    if z_eig is None:
        raise ValueError("z_eig is required to compute the full many-body z shift vector.")

    n_exc = A_coeff.shape[1]
    A = A_coeff.reshape(Nk, Nv, Nc, n_exc)
    z_cc = z_eig[:, c_idx, :][:, :, c_idx]
    z_vv = z_eig[:, v_idx, :][:, :, v_idx]

    conduction = np.einsum('kvps,kvcs,kpc->s',
                           A.conj(), A, z_cc, optimize=True)
    valence = np.einsum('kpcs,kvcs,kvp->s',
                        A.conj(), A, z_vv, optimize=True)
    return np.real(conduction - valence)


def _many_body_inplane_shift_vectors(A_coeff, evecs, v_idx, c_idx, n_k_side, dk):
    r"""
    R^x_S0 and R^y_S0 from Eq. (7) of ref.pdf on the BSE k-grid.

    For z, the position operator is an ordinary layer-space matrix.  For x/y,
    the position operator in a periodic crystal is represented by a Berry
    connection in k space, so we evaluate the many-body shift vector through
    finite differences on neighboring k points.

    The direct-transition BSE basis state is

        |v c k> = |u_c(k)>_electron x |u_v(k)>^*_hole .

    Its nearest-neighbor overlap is

        L_S(k,k+dk) =
            A^*_S(v,c,k) A_S(v',c',k+dk)
            <u_c,k|u_c',k+dk> <u_v',k+dk|u_v,k>

    with all repeated valence/conduction indices summed.  Expanding this link
    to first order in dk gives the same ingredients as Eq. (7): the conduction
    Berry connection, minus the valence Berry connection, plus the k-derivative
    of the exciton envelope A^S_vck.  Contracting the full link instead of
    separate band phases keeps the result invariant under k-dependent phase or
    unitary rotations of the active bands, up to the active-space truncation.
    """
    Nk = int(n_k_side) * int(n_k_side)
    Nv = len(v_idx)
    Nc = len(c_idx)
    dim_bse = Nk * Nv * Nc
    if A_coeff.shape[0] != dim_bse:
        raise ValueError("A_coeff size is inconsistent with n_k_side.")
    if n_k_side < 2:
        raise ValueError("At least two k points per direction are required.")
    if dk <= 0.0:
        raise ValueError("dk must be positive.")

    n_exc = A_coeff.shape[1]
    A = A_coeff.reshape(Nk, Nv, Nc, n_exc)
    U_c = evecs[:, :, c_idx]
    U_v = evecs[:, :, v_idx]

    def flat_index(ix, iy):
        return iy * n_k_side + ix

    def exciton_link(k0, k1):
        O_c = U_c[k0].conj().T @ U_c[k1]
        O_h = U_v[k1].conj().T @ U_v[k0]
        return np.einsum('vcs,pv,cq,pqs->s',
                         A[k0].conj(), O_h, O_c, A[k1], optimize=True)

    def link_forward(k0, k1):
        return -np.imag(exciton_link(k0, k1)) / dk

    def link_backward(k0, k1):
        return np.imag(exciton_link(k0, k1)) / dk

    R = {'x': np.zeros(n_exc, dtype=np.float64),
         'y': np.zeros(n_exc, dtype=np.float64)}

    for direction in ['x', 'y']:
        for iy in range(n_k_side):
            for ix in range(n_k_side):
                k0 = flat_index(ix, iy)

                if direction == 'x':
                    has_plus = ix + 1 < n_k_side
                    has_minus = ix - 1 >= 0
                    k_plus = flat_index(ix + 1, iy) if has_plus else None
                    k_minus = flat_index(ix - 1, iy) if has_minus else None
                else:
                    has_plus = iy + 1 < n_k_side
                    has_minus = iy - 1 >= 0
                    k_plus = flat_index(ix, iy + 1) if has_plus else None
                    k_minus = flat_index(ix, iy - 1) if has_minus else None

                if has_plus and has_minus:
                    R[direction] += 0.5 * (
                        link_forward(k0, k_plus) + link_backward(k0, k_minus)
                    )
                elif has_plus:
                    R[direction] += link_forward(k0, k_plus)
                elif has_minus:
                    R[direction] += link_backward(k0, k_minus)

    return R


def _ipa_inplane_shift_current_from_pipeline(pipe, omegas, eta):
    """IPA in-plane shift current on the same active space as a BSE pipeline."""
    comp_list = [('x', 'x', 'x'), ('x', 'y', 'y'),
                 ('y', 'x', 'x'), ('y', 'y', 'y')]

    evals = pipe['evals']
    Nb = pipe['Nb']
    Nk = pipe['Nk']
    v_idx = pipe['v_idx']
    c_idx = pipe['c_idx']
    A_uc = pipe['A_uc']

    v_map = {'x': pipe['vx_eig'], 'y': pipe['vy_eig']}
    w_map = {
        'xx': pipe['w_xx_eig'],
        'yy': pipe['w_yy_eig'],
        'xy': pipe['w_xy_eig'],
        'yx': pipe['w_xy_eig'],
    }

    eps_denom = 1e-5
    mid = Nb // 2
    w_all = evals[:, :, None] - evals[:, None, :]
    results = {}

    for comp in comp_list:
        a_dir, b_dir, c_dir = comp
        v_a = v_map[a_dir]
        v_b = v_map[b_dir]
        v_c = v_map[c_dir]
        w_ac = w_map[a_dir + c_dir]
        sigma = np.zeros_like(omegas)

        for n in c_idx:
            f_n = 1.0 if n < mid else 0.0
            v_c_n_row = v_c[:, n, :]
            v_a_n_row = v_a[:, n, :]
            v_a_nn = v_a[:, n, n]
            v_c_nn = v_c[:, n, n]
            w_n_all = w_all[:, n, :]

            for m in v_idx:
                f_m = 1.0 if m < mid else 0.0
                f_nm = f_n - f_m
                if f_nm == 0.0:
                    continue

                w_nm = evals[:, n] - evals[:, m]
                nonzero = w_nm > eps_denom

                r_b_mn = np.zeros(Nk, dtype=np.complex128)
                r_b_mn[nonzero] = v_b[nonzero, m, n] / (-1j * w_nm[nonzero])

                termA = np.zeros(Nk, dtype=np.complex128)
                delta_a = v_a_nn[nonzero] - v_a[nonzero, m, m]
                delta_c = v_c_nn[nonzero] - v_c[nonzero, m, m]
                termA[nonzero] = (
                    v_c[nonzero, n, m] * delta_a
                    + v_a[nonzero, n, m] * delta_c
                ) / w_nm[nonzero]

                w_np = w_n_all
                w_pm = w_all[:, :, m]
                valid_p = (np.abs(w_np) > eps_denom) & (np.abs(w_pm) > eps_denom)
                valid_p[:, n] = False
                valid_p[:, m] = False
                valid_p &= nonzero[:, None]

                v_a_col_m = v_a[:, :, m]
                v_c_col_m = v_c[:, :, m]
                num1 = v_c_n_row * v_a_col_m
                num2 = v_a_n_row * v_c_col_m
                termB_contrib = np.zeros((Nk, Nb), dtype=np.complex128)
                termB_contrib[valid_p] = (
                    num1[valid_p] / w_pm[valid_p]
                    - num2[valid_p] / w_np[valid_p]
                )
                termB = np.sum(termB_contrib, axis=1)

                termC = -w_ac[:, n, m]
                K_nm = termA + termB + termC
                r_deriv = np.zeros(Nk, dtype=np.complex128)
                r_deriv[nonzero] = K_nm[nonzero] / (-1j * w_nm[nonzero])

                weight = f_nm * np.imag(r_b_mn * r_deriv)
                diff = omegas[:, None] - w_nm[None, :]
                lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
                sigma += np.sum(lorentz * weight[None, :], axis=1)

        sigma /= Nk
        e_charge = 1.602176634e-19
        hbar = 1.054571817e-34
        prefactor = (2 * np.pi * e_charge**2) / (hbar * A_uc) * 1E6
        results[comp] = sigma * prefactor

    return results


def calculate_bse_z_shift_current(N_top=1, N_bottom=1, twist_angle=0.0,
                                   E_range=(0.0, 1.0), n_E=400, eta=0.010,
                                   k_range=0.15, n_k_bse=30,
                                   n_val=2, n_cond=2,
                                   thickness=5.2,
                                   kappa=2.5, r0=5.0,
                                   plot_ipa_comparison=True,
                                   band_window=None, save_prefix="",
                                   use_gpu="auto", gpu_dtype="complex64",
                                   gpu_full_eigh_max_dim=32000):
    r"""
    Excitonic z-shift current via the Bethe-Salpeter equation (BSE).
    Ref: Lai, M., Xuan, F. & Quek, S. Y. arXiv:2402.02002

    Solves the BSE to obtain exciton wavefunctions, then computes:

        sigma^{z;bb}(ω) = C * Σ_S R^z_{S0} * |d^b_S|^2 * δ(Ω_S - ω)

    where:
        d^b_S = Σ_{vck} A^S_{vck} r^b_{cv}(k)    (exciton optical dipole)
        R^z_{S0} = Σ_{vck} |A^S_{vck}|^2 Δz(vck)  (Many-body shift vector)
        Integrand = R^z_{S0} * |d^b_S|^2

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
    print(f"  Keldysh params: kappa={kappa}, r0={r0} A")
    print(f"  thickness={thickness} A")

    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    print(f"\n[1-4] Running BSE pipeline...")
    pipe = _run_bse_pipeline(model, k_range, n_k_bse, n_val, n_cond,
                             thickness=thickness, kappa=kappa, r0=r0,
                             band_window=band_window, use_gpu=use_gpu,
                             gpu_dtype=gpu_dtype,
                             gpu_full_eigh_max_dim=gpu_full_eigh_max_dim)

    Nk = pipe['Nk']
    dE = pipe['dE']
    r_b = pipe['r_b']
    z_eig = pipe['z_eig']
    delta_z = pipe['delta_z']
    Omega_S = pipe['Omega_S']
    A_coeff = pipe['A_coeff']
    dim_bse = pipe['dim_bse']
    v_idx = pipe['v_idx']
    c_idx = pipe['c_idx']
    Nv, Nc = pipe['Nv'], pipe['Nc']

    a_lat = model.a_lat
    b_lat = model.b_lat
    A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2
    V_uc = A_uc * thickness * (N_top + N_bottom)

    delta_z_flat = delta_z.reshape(dim_bse)
    omegas = np.linspace(E_range[0], E_range[1], n_E)
    results = {}
    # R^z_{S0}: Many-body shift vector
    R_z_S0 = _many_body_z_shift_vector(A_coeff, z_eig, v_idx, c_idx, Nk, Nv, Nc)

    print(f"\n[5] Computing exciton observables...")
    for comp in comp_list:
        a_dir, b_dir, c_dir = comp
        assert b_dir == c_dir, "z-shift current only for linearly polarized light (b==c)"
        print(f"  sigma^{{{a_dir}{b_dir}{c_dir}}}...")

        r_b_flat = r_b[b_dir].reshape(dim_bse)

        d_b_S = A_coeff.conj().T @ r_b_flat
        
        integrand_S = R_z_S0 * np.abs(d_b_S)**2
        
        print(f"    Lowest exciton optical dipole |d^{b_dir}_0|: {np.abs(d_b_S[0]):.6e}")

        bse_sum = np.sum(integrand_S)
        ipa_sum = np.sum(np.real(delta_z_flat * np.abs(r_b_flat)**2))
        print(f"    Shift weight sum check {a_dir}{b_dir}{c_dir}: BSE={bse_sum:.6e}, IPA={ipa_sum:.6e}, "
                f"enhancement ratio={bse_sum/ipa_sum:.6f}")

        diff = omegas[:, None] - Omega_S[None, :]
        lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
        sigma = lorentz @ integrand_S / Nk
        results[comp] = sigma

    e_charge = 1.602176634e-19
    hbar = 1.054571817e-34
    prefactor = (2 * np.pi * e_charge**2) / (hbar * V_uc) * 1E6
    for comp in comp_list:
        results[comp] *= prefactor

    ipa_results = None
    if plot_ipa_comparison:
        print(f"\n[6] Computing IPA comparison on same grid...")
        ipa_results = {}
        for comp in comp_list:
            a_dir, b_dir, c_dir = comp
            r_b_flat = r_b[b_dir].reshape(dim_bse)
            rb_sq = np.abs(r_b_flat)**2
            integrand_ipa = delta_z.reshape(dim_bse) * rb_sq

            diff = omegas[:, None] - dE.reshape(dim_bse)[None, :]
            lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
            ipa_results[comp] = lorentz @ integrand_ipa / Nk * prefactor

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = {('z', 'x', 'x'): (r'$\sigma^{zxx}$', 'r'),
              ('z', 'y', 'y'): (r'$\sigma^{zyy}$', 'b')}
    for ax, comp in zip(axes, comp_list):
        lbl, col = labels[comp]
        ax.plot(omegas, results[comp], color=col, lw=2, label=f'BSE {lbl}')
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
                 f'kappa={kappa}, r0={r0} A, d={thickness} A, '
                 f'grid={n_k_bse}²', fontsize=11)
    plt.tight_layout()
    fname = f"EM_bse_z_sc{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved BSE Z-Shift Current Figure: {fname}")
    spec_cols = [
        omegas,
        results[('z', 'x', 'x')],
        results[('z', 'y', 'y')],
    ]
    spec_header = [
        "omega_eV",
        "BSE_sigma_zxx_uA_per_V2",
        "BSE_sigma_zyy_uA_per_V2",
    ]
    if ipa_results is not None:
        spec_cols.extend([
            ipa_results[('z', 'x', 'x')],
            ipa_results[('z', 'y', 'y')],
        ])
        spec_header.extend([
            "IPA_sigma_zxx_uA_per_V2",
            "IPA_sigma_zyy_uA_per_V2",
        ])
    _save_dat(
        f"EM_bse_z_sc{save_prefix}.dat",
        np.column_stack(spec_cols),
        " ".join(spec_header)
    )

    # Exciton analysis plot
    osc_data = {}
    shift_weight_data = {}
    for comp in comp_list:
        b_dir = comp[1]
        r_b_flat = r_b[b_dir].reshape(dim_bse)
        d_b_S = A_coeff.conj().T @ r_b_flat
        osc_data[b_dir] = np.abs(d_b_S)**2
        shift_weight_data[b_dir] = R_z_S0 * np.abs(d_b_S)**2

    plot_exciton_analysis(Omega_S, osc_data, shift_weight_data, E_range=E_range,
                          N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle,
                          kappa=kappa, r0=r0, thickness=thickness,
                          save_prefix=save_prefix)

    return omegas, results, Omega_S, A_coeff


def calculate_bse_inplane_shift_current(N_top=1, N_bottom=1, twist_angle=0.0,
                                        E_range=(0.0, 1.0), n_E=400, eta=0.010,
                                        k_range=0.15, n_k_bse=30,
                                        n_val=2, n_cond=2,
                                        kappa=2.5, r0=5.0,
                                        plot_ipa_comparison=True,
                                        band_window=None, save_prefix="",
                                        use_gpu="auto", gpu_dtype="complex128",
                                        gpu_full_eigh_max_dim=32000):
    r"""
    Excitonic in-plane shift current via BSE in length gauge.

    Computes the linearly polarized in-plane components

        sigma^{x;xx}, sigma^{x;yy}, sigma^{y;xx}, sigma^{y;yy}

    using Eq. (5) and Eq. (7) of ref.pdf:

        sigma^{a;bb}(omega) = C * sum_S R^a_{S0} |d^b_S|^2
                              delta(Omega_S - omega)

    where d^b_S is the BSE optical dipole and R^a_{S0} is evaluated from the
    many-body position expectation value on the BSE k-grid.  Because x and y
    are periodic directions, R^a_{S0} is calculated with gauge-covariant
    nearest-neighbor links of both the Bloch functions and the exciton envelope.

    The returned spectra use the same 2D units as calculate_shift_current:
    microampere * Angstrom / V^2.

    The default BSE dtype is complex128 because symmetry-forbidden in-plane
    components are obtained by cancellation; complex64 can leave visible
    numerical residuals even when the exact response is zero.
    """
    comp_list = [('x', 'x', 'x'), ('x', 'y', 'y'),
                 ('y', 'x', 'x'), ('y', 'y', 'y')]

    print("=" * 60)
    print("BSE Excitonic In-Plane Shift Current Calculation (Effective Model)")
    print("=" * 60)
    print(f"  Grid: {n_k_bse}x{n_k_bse} = {n_k_bse**2} k-points")
    print(f"  Active space: {n_val}v x {n_cond}c")
    print(f"  BSE dimension: {n_val * n_cond * n_k_bse**2}")
    print(f"  Keldysh params: kappa={kappa}, r0={r0} A")

    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    print(f"\n[1-4] Running BSE pipeline...")
    pipe = _run_bse_pipeline(model, k_range, n_k_bse, n_val, n_cond,
                             thickness=None, kappa=kappa, r0=r0,
                             band_window=band_window, use_gpu=use_gpu,
                             gpu_dtype=gpu_dtype,
                             gpu_full_eigh_max_dim=gpu_full_eigh_max_dim)

    Nk = pipe['Nk']
    r_b = pipe['r_b']
    Omega_S = pipe['Omega_S']
    A_coeff = pipe['A_coeff']
    dim_bse = pipe['dim_bse']
    A_uc = pipe['A_uc']

    omegas = np.linspace(E_range[0], E_range[1], n_E)
    results = {}

    d_S = {}
    osc_S = {}
    for b_dir in ['x', 'y']:
        r_b_flat = r_b[b_dir].reshape(dim_bse)
        d_S[b_dir] = A_coeff.conj().T @ r_b_flat
        osc_S[b_dir] = np.abs(d_S[b_dir])**2
        print(f"    Lowest exciton optical dipole |d^{b_dir}_0|: {np.abs(d_S[b_dir][0]):.6e}")

    print(f"\n[5] Computing many-body in-plane shift vectors...")
    R_inplane = _many_body_inplane_shift_vectors(
        A_coeff, pipe['evecs'], pipe['v_idx'], pipe['c_idx'],
        n_k_side=n_k_bse, dk=pipe['dk'],
    )
    print(f"    Lowest exciton R^x_S0: {R_inplane['x'][0]:.6e} A")
    print(f"    Lowest exciton R^y_S0: {R_inplane['y'][0]:.6e} A")

    print(f"\n[6] Computing excitonic in-plane shift-current spectra...")
    diff = omegas[:, None] - Omega_S[None, :]
    lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)

    for comp in comp_list:
        a_dir, b_dir, c_dir = comp
        assert b_dir == c_dir, "Only linearly polarized components with b==c are implemented."
        integrand_S = R_inplane[a_dir] * osc_S[b_dir]
        sigma = lorentz @ integrand_S / Nk
        results[comp] = sigma
        print(f"    sigma^{{{a_dir}{b_dir}{c_dir}}}: "
              f"sum(R|d|^2)={np.sum(integrand_S):.6e}")

    e_charge = 1.602176634e-19
    hbar = 1.054571817e-34
    prefactor = (2 * np.pi * e_charge**2) / (hbar * A_uc) * 1E6
    for comp in comp_list:
        results[comp] *= prefactor
    print("    BSE max |sigma| after prefactor:")
    for comp in comp_list:
        a_dir, b_dir, c_dir = comp
        print(f"      {a_dir}{b_dir}{c_dir}: {np.max(np.abs(results[comp])):.6e} uA*A/V^2")

    ipa_results = None
    if plot_ipa_comparison:
        print(f"\n[7] Computing IPA comparison on same active-space grid...")
        ipa_results = _ipa_inplane_shift_current_from_pipeline(pipe, omegas, eta)
        print("    IPA max |sigma| on the same grid:")
        for comp in comp_list:
            a_dir, b_dir, c_dir = comp
            print(f"      {a_dir}{b_dir}{c_dir}: {np.max(np.abs(ipa_results[comp])):.6e} uA*A/V^2")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    plot_info = {
        ('x', 'x', 'x'): (axes[0, 0], r'$\sigma^{xxx}$', 'tab:red'),
        ('x', 'y', 'y'): (axes[0, 1], r'$\sigma^{xyy}$', 'tab:purple'),
        ('y', 'x', 'x'): (axes[1, 0], r'$\sigma^{yxx}$', 'tab:green'),
        ('y', 'y', 'y'): (axes[1, 1], r'$\sigma^{yyy}$', 'tab:blue'),
    }
    for comp in comp_list:
        ax, lbl, col = plot_info[comp]
        ax.plot(omegas, results[comp], color=col, lw=2, label=f'BSE {lbl}')
        if ipa_results is not None:
            ax.plot(omegas, ipa_results[comp], color=col, lw=1.5, ls='--',
                    alpha=0.6, label=f'IPA {lbl}')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_ylabel(r'$\mu$A$\cdot$A/V$^2$')
        ax.set_title(lbl)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(E_range)
    axes[1, 0].set_xlabel('Photon Energy (eV)')
    axes[1, 1].set_xlabel('Photon Energy (eV)')

    fig.suptitle(f'Excitonic In-Plane Shift Current (BSE, Length Gauge)\n'
                 f'N={N_top}/{N_bottom}, twist={np.degrees(twist_angle):.0f} deg, '
                 f'kappa={kappa}, r0={r0} A, grid={n_k_bse}^2', fontsize=11)
    plt.tight_layout()
    fname = f"EM_bse_inplane_sc{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved BSE In-Plane Shift Current Figure: {fname}")

    spec_cols = [
        omegas,
        results[('x', 'x', 'x')],
        results[('x', 'y', 'y')],
        results[('y', 'x', 'x')],
        results[('y', 'y', 'y')],
    ]
    spec_header = [
        "omega_eV",
        "BSE_sigma_xxx_uA_A_per_V2",
        "BSE_sigma_xyy_uA_A_per_V2",
        "BSE_sigma_yxx_uA_A_per_V2",
        "BSE_sigma_yyy_uA_A_per_V2",
    ]
    if ipa_results is not None:
        spec_cols.extend([
            ipa_results[('x', 'x', 'x')],
            ipa_results[('x', 'y', 'y')],
            ipa_results[('y', 'x', 'x')],
            ipa_results[('y', 'y', 'y')],
        ])
        spec_header.extend([
            "IPA_sigma_xxx_uA_A_per_V2",
            "IPA_sigma_xyy_uA_A_per_V2",
            "IPA_sigma_yxx_uA_A_per_V2",
            "IPA_sigma_yyy_uA_A_per_V2",
        ])
    _save_dat(
        f"EM_bse_inplane_sc{save_prefix}.dat",
        np.column_stack(spec_cols),
        " ".join(spec_header)
    )

    exciton_cols = [
        Omega_S,
        osc_S['x'], osc_S['y'],
        R_inplane['x'], R_inplane['y'],
        R_inplane['x'] * osc_S['x'],
        R_inplane['x'] * osc_S['y'],
        R_inplane['y'] * osc_S['x'],
        R_inplane['y'] * osc_S['y'],
    ]
    _save_dat(
        f"EM_bse_inplane_exciton_weights{save_prefix}.dat",
        np.column_stack(exciton_cols),
        "exciton_energy_eV osc_x osc_y R_x_A R_y_A "
        "weight_xxx weight_xyy weight_yxx weight_yyy"
    )

    return omegas, results, Omega_S, A_coeff


def calculate_bse_absorbance(N_top=1, N_bottom=1, twist_angle=0.0,
                              E_range=(0.0, 1.0), n_E=500, eta=0.100,
                              k_range=0.15, n_k_bse=30,
                              n_val=2, n_cond=2,
                              kappa=2.5, r0=5.0,
                              plot_ipa_comparison=True,
                              band_window=None, save_prefix="",
                              use_gpu="auto", gpu_dtype="complex64",
                              gpu_full_eigh_max_dim=32000):
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
    print(f"  Keldysh params: kappa={kappa}, r0={r0} A")

    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    print(f"\n[1-4] Running BSE pipeline...")
    pipe = _run_bse_pipeline(model, k_range, n_k_bse, n_val, n_cond,
                             thickness=None, kappa=kappa, r0=r0,
                             band_window=band_window, use_gpu=use_gpu,
                             gpu_dtype=gpu_dtype,
                             gpu_full_eigh_max_dim=gpu_full_eigh_max_dim)

    Nk = pipe['Nk']
    dE = pipe['dE']
    r_b = pipe['r_b']
    Omega_S = pipe['Omega_S']
    A_coeff = pipe['A_coeff']
    dim_bse = pipe['dim_bse']

    omegas = np.linspace(E_range[0], E_range[1], n_E)
    abs_bse = {}

    print(f"\n[5] Computing BSE absorbance spectrum...")
    for b_dir in ['x', 'y']:
        r_b_flat = r_b[b_dir].reshape(dim_bse)
        d_b_S = A_coeff.conj().T @ r_b_flat
        osc_S = np.abs(d_b_S)**2

        bse_osc_sum = np.sum(osc_S)
        ipa_osc_sum = np.sum(np.abs(r_b_flat)**2)
        print(f"    F-sum check ({b_dir}-pol): BSE={bse_osc_sum:.6e}, IPA={ipa_osc_sum:.6e}, "
                f"ratio={bse_osc_sum/ipa_osc_sum:.6f}")

        diff = omegas[:, None] - Omega_S[None, :]
        lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
        abs_bse[b_dir] = (lorentz @ osc_S / Nk) * omegas

    abs_ipa = None
    if plot_ipa_comparison:
        print(f"\n[6] Computing IPA comparison on same grid...")
        abs_ipa = {}
        dE_flat = dE.reshape(dim_bse)
        for b_dir in ['x', 'y']:
            rb_sq = np.abs(r_b[b_dir].reshape(dim_bse))**2
            diff = omegas[:, None] - dE_flat[None, :]
            lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
            abs_ipa[b_dir] = (lorentz @ rb_sq / Nk) * omegas

    # Plotting
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
              f'kappa={kappa}, r0={r0} A, grid={n_k_bse}²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)

    fname = f"EM_bse_absorbance{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved BSE Absorbance Figure: {fname}")
    spec_cols = [omegas, abs_bse['x'], abs_bse['y']]
    spec_header = ["omega_eV", "BSE_abs_x", "BSE_abs_y"]
    if abs_ipa is not None:
        spec_cols.extend([abs_ipa['x'], abs_ipa['y']])
        spec_header.extend(["IPA_abs_x", "IPA_abs_y"])
    _save_dat(
        f"EM_bse_absorbance{save_prefix}.dat",
        np.column_stack(spec_cols),
        " ".join(spec_header)
    )

    return omegas, abs_bse, Omega_S, A_coeff


def plot_exciton_oscillator_strength(N_top=1, N_bottom=1, twist_angle=0.0,
                                      E_range=(0.0, 1.0), eta=0.010,
                                      k_range=0.15, n_k_bse=30,
                                      n_val=2, n_cond=2,
                                      kappa=2.5, r0=5.0,
                                      polarization='x',
                                      n_show=100,
                                      plot_broadened=True,
                                      band_window=None, save_prefix="",
                                      use_gpu="auto", gpu_dtype="complex64",
                                      gpu_full_eigh_max_dim=32000):
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
    print(f"  Keldysh params: kappa={kappa}, r0={r0} A")

    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    print(f"\n[1-4] Running BSE pipeline...")
    pipe = _run_bse_pipeline(model, k_range, n_k_bse, n_val, n_cond,
                             thickness=None, kappa=kappa, r0=r0,
                             band_window=band_window, use_gpu=use_gpu,
                             gpu_dtype=gpu_dtype,
                             gpu_full_eigh_max_dim=gpu_full_eigh_max_dim)

    Nk = pipe['Nk']
    Omega_S = pipe['Omega_S']
    A_coeff = pipe['A_coeff']
    r_b = pipe['r_b']
    dim_bse = pipe['dim_bse']
    # r_x_flat, r_y_flat, and degenerate resolution already done by the pipeline

    # Compute oscillator strength |d^b_S|^2 for each exciton
    osc = {}
    for b_dir in ['x', 'y']:
        r_flat = r_b[b_dir].reshape(dim_bse)
        d_S = A_coeff.conj().T @ r_flat
        osc[b_dir] = np.abs(d_S)**2 / Nk

    # Select excitons in energy range
    mask = (Omega_S >= E_range[0]) & (Omega_S <= E_range[1])
    idx = np.where(mask)[0][:n_show]
    E_sel = Omega_S[idx]

    pol_list = ['x', 'y'] if polarization == 'both' else [polarization]

    for b_dir in pol_list:
        osc_sel = osc[b_dir][idx]
        bright_order = np.argsort(osc_sel)[::-1]
        print(f"\n  Top 5 brightest excitons ({b_dir}-pol):")
        for rank, si in enumerate(bright_order[:5]):
            print(f"    #{rank+1}: E = {E_sel[si]:.4f} eV, "
                  f"|d^{b_dir}|² = {osc_sel[si]:.4e}")

    # Plot
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
                          f'kappa={kappa}, r0={r0} A, grid={n_k_bse}²')
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
                     f'kappa={kappa}, r0={r0} A, grid={n_k_bse}²')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(E_range)

    plt.tight_layout()
    fname = f"EM_exciton_osc_strength_{polarization}{save_prefix}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved: {fname}")
    _save_dat(
        f"EM_exciton_osc_strength_{polarization}{save_prefix}.dat",
        np.column_stack([Omega_S, osc['x'], osc['y']]),
        "exciton_energy_eV osc_x_per_Nk osc_y_per_Nk"
    )
    if plot_broadened:
        omegas_env = np.linspace(E_range[0], E_range[1], 500)
        diff_env = omegas_env[:, None] - Omega_S[None, :]
        lorentz_env = (1.0 / np.pi) * eta / (diff_env**2 + eta**2)
        env_x = lorentz_env @ osc['x']
        env_y = lorentz_env @ osc['y']
        _save_dat(
            f"EM_exciton_osc_strength_{polarization}_envelope{save_prefix}.dat",
            np.column_stack([omegas_env, env_x, env_y]),
            "omega_eV envelope_x envelope_y"
        )

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
    _save_dat(
        f"EM_bse_exciton_analysis{save_prefix}.dat",
        np.column_stack([
            E_sel,
            osc_data['x'][idx],
            osc_data['y'][idx],
            shift_weight_data['x'][idx],
            shift_weight_data['y'][idx],
        ]),
        "exciton_energy_eV osc_x osc_y shift_weight_x shift_weight_y"
    )


def _resolve_degenerate_excitons(Omega_S, A_coeff, r_b_x_flat, r_b_y_flat,
                                  degen_tol=1e-5):
    """
    Rotate degenerate exciton subspaces so that each state is an eigenstate
    of polarization (x-bright or y-bright), rather than an arbitrary mixture.

    Two-step procedure within each degenerate multiplet:
      1. Diagonalize D^x = d_x d_x^H to isolate the x-bright direction.
      2. Within the x-dark subspace, diagonalize D^y to resolve y-bright states.

    After rotation (for deg >= 2):
      - Largest  x-oscillator state is last (index j-1).
      - Largest  y-oscillator state is next-to-last (index j-2) and x-dark.
      - Remaining states are both x-dark and y-dark (when deg > 2).

    Parameters
    ----------
    degen_tol : float
        Energy tolerance (eV) for grouping excitons as degenerate.

    Returns
    -------
    A_coeff_rot : ndarray
        Rotated BSE eigenvectors (same shape as A_coeff).
    """
    A_rot = A_coeff.copy()

    i = 0
    while i < len(Omega_S):
        j = i + 1
        while j < len(Omega_S) and abs(Omega_S[j] - Omega_S[i]) < degen_tol:
            j += 1
        deg = j - i
        if deg > 1:
            # Step 1: diagonalize D^x to separate x-bright from x-dark
            A_sub = A_rot[:, i:j]  # (dim_bse, deg)
            d_x = A_sub.conj().T @ r_b_x_flat  # (deg,)
            _, U_x = np.linalg.eigh(np.outer(d_x, d_x.conj()))
            # eigh returns ascending: [0, ..., 0, |d_x|^2]
            A_sub = A_sub @ U_x
            A_rot[:, i:j] = A_sub

            # Step 2: within x-dark subspace, resolve y-bright states
            n_x_dark = deg - 1
            if n_x_dark > 1:
                d_y_dark = A_sub[:, :n_x_dark].conj().T @ r_b_y_flat  # (n_x_dark,)
                _, U_y = np.linalg.eigh(np.outer(d_y_dark, d_y_dark.conj()))
                A_rot[:, i:i + n_x_dark] = A_sub[:, :n_x_dark] @ U_y
        i = j

    return A_rot


def analyze_exciton_wavefunction(N_top=1, N_bottom=1, twist_angle=0.0,
                                  E_range=(0.0, 1.0), eta=0.010,
                                  k_range=0.15, n_k_bse=30,
                                  n_val=2, n_cond=2,
                                  thickness=5.2,
                                  kappa=2.5, r0=5.0,
                                  n_excitons=4,
                                  band_window=None, save_prefix="",
                                  use_gpu="auto", gpu_dtype="complex64",
                                  gpu_full_eigh_max_dim=32000):
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

    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

    print(f"[1-2] Running BSE pipeline...")
    pipe = _run_bse_pipeline(model, k_range, n_k_bse, n_val, n_cond,
                             thickness=thickness, kappa=kappa, r0=r0,
                             band_window=band_window, use_gpu=use_gpu,
                             gpu_dtype=gpu_dtype,
                             gpu_full_eigh_max_dim=gpu_full_eigh_max_dim)

    Nk = pipe['Nk']
    Nb = pipe['Nb']
    evals = pipe['evals']
    evecs = pipe['evecs']
    v_idx = pipe['v_idx']
    c_idx = pipe['c_idx']
    Nv, Nc = pipe['Nv'], pipe['Nc']
    dE = pipe['dE']
    r_b = pipe['r_b']
    r_x_flat = pipe['r_x_flat']
    r_y_flat = pipe['r_y_flat']
    delta_z = pipe['delta_z']  # (Nk, Nv, Nc)
    KX, KY = pipe['KX'], pipe['KY']
    dk = pipe['dk']
    Omega_S = pipe['Omega_S']
    A_coeff = pipe['A_coeff']
    dim_bse = pipe['dim_bse']

    # Recompute dipoles with rotated coefficients
    d_x_all = A_coeff.conj().T @ r_x_flat
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
    weight_rows = []

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

        # --- Column 1: Real-space envelope |Φ(r)|² / Shift Vector ---
        # psi_r_sq = np.zeros((n_k_bse, n_k_bse))
        # for iv in range(Nv):
        #     for ic in range(Nc):
        #         phi_k = A_4d[:, :, iv, ic]
        #         phi_r = np.fft.fftshift(np.fft.fft2(phi_k))
        #         psi_r_sq += np.abs(phi_r)**2
        # psi_r_sq /= np.max(psi_r_sq)

        # ax = axes[row, 1]
        # im = ax.pcolormesh(RX, RY, psi_r_sq, cmap='inferno', shading='auto')
        # fig.colorbar(im, ax=ax, shrink=0.8)
        # ax.set_xlabel(r'$\Delta x$ (Å)')
        # ax.set_ylabel(r'$\Delta y$ (Å)')
        # ax.set_title(f'S{row} ({pol}): Real-space envelope\n'
        #              r'$|\Phi^S(\mathbf{r}_e - \mathbf{r}_h)|^2$')
        # ax.set_aspect('equal')
        
        # New Column 1: Shift Vector in k-space
        # delta_z = z_c - z_v. Shift vector is -z_c + z_v = -delta_z
        # We weight it by the exciton coefficients |A^S_{vck}|^2
        shift_vector_k = np.sum(np.abs(A_3d)**2 * (-delta_z), axis=(1, 2))  # (Nk,)
        shift_vector_k = shift_vector_k.reshape(n_k_bse, n_k_bse)

        ax = axes[row, 1]
        im = ax.pcolormesh(KX, KY, shift_vector_k, cmap='coolwarm', shading='auto')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (Å$^{-1}$)')
        ax.set_title(f'S{row} ({pol}): Shift Vector\n'
                     r'$\sum_{vc}|A^S_{vck}|^2 (-\langle u_c|z|u_c\rangle + \langle u_v|z|u_v\rangle)$')
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
        weight_rows.append(
            [row, int(si), Omega_S[si], osc_x[si], osc_y[si],
             1.0 if osc_x[si] > osc_y[si] else 0.0,
             rho_e_top, rho_e_bot, rho_h_top, rho_h_bot]
            + list(rho_e) + list(rho_h)
        )

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
    orbital_headers = (
        [f"rho_e_orb{i}" for i in range(Nb)]
        + [f"rho_h_orb{i}" for i in range(Nb)]
    )
    _save_dat(
        f"EM_exciton_wavefunction_weights{save_prefix}.dat",
        np.array(weight_rows, dtype=float),
        "rank exciton_index exciton_energy_eV osc_x osc_y pol_is_x "
        "rho_e_top rho_e_bottom rho_h_top rho_h_bottom "
        + " ".join(orbital_headers)
    )

    return Omega_S, A_coeff, bright_idx


def study_x_exciton_dipole_vs_shift_peak(layer_pairs=None, N_layers=(2, 3),
                                         twist_angle=0.0,
                                         E_range=(0.0, 1.0), n_E=500, eta=0.010,
                                         k_range=0.15, n_k_bse=30,
                                         n_val=2, n_cond=2,
                                         thickness=5.2,
                                         kappa=2.5, r0=5.0,
                                         band_window=None, save_prefix="",
                                         use_gpu="auto", gpu_dtype="complex64",
                                         gpu_full_eigh_max_dim=32000):
    r"""
    Scan different (N_top, N_bottom) stacks and correlate:
      1) dipole of the lowest-energy x-bright exciton
      2) first peak of sigma^{zxx}(omega)

    Exciton dipole is extracted from layer-resolved electron/hole densities:
        p_z = <z_e> - <z_h>
    where <z_e/h> are computed from the same weights used in
    `analyze_exciton_wavefunction`.

    Parameters
    ----------
    layer_pairs : list[tuple[int, int]] or None
        Explicit (N_top, N_bottom) list. Example: [(2, 2), (3, 3)].
        If None, uses symmetric pairs from `N_layers`: [(N, N), ...].
    N_layers : iterable[int]
        Layer counts used when `layer_pairs` is None.
    """
    if layer_pairs is None:
        layer_pairs = [(int(n), int(n)) for n in N_layers]
    if len(layer_pairs) == 0:
        raise ValueError("layer_pairs is empty.")

    def _first_peak_idx(x, y):
        y_abs = np.abs(y)
        if len(y_abs) < 3:
            return int(np.argmax(y_abs))
        cand = np.where((y_abs[1:-1] > y_abs[:-2]) & (y_abs[1:-1] >= y_abs[2:]))[0] + 1
        if len(cand) == 0:
            return int(np.argmax(y_abs))
        return int(cand[0])

    results = []

    print("=" * 68)
    print("Scan: Lowest x-bright exciton dipole vs first zxx shift-current peak")
    print("=" * 68)

    for N_top, N_bottom in layer_pairs:
        print(f"\n--- Case N_top/N_bottom = {N_top}/{N_bottom} ---")

        model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)

        pipe = _run_bse_pipeline(model, k_range, n_k_bse, n_val, n_cond,
                                 thickness=thickness, kappa=kappa, r0=r0,
                                 band_window=band_window, use_gpu=use_gpu,
                                 gpu_dtype=gpu_dtype,
                                 gpu_full_eigh_max_dim=gpu_full_eigh_max_dim)

        Nk = pipe['Nk']
        Nb = pipe['Nb']
        evecs = pipe['evecs']
        v_idx = pipe['v_idx']
        c_idx = pipe['c_idx']
        Nv, Nc = pipe['Nv'], pipe['Nc']
        U_dag = pipe['U_dag']
        Omega_S = pipe['Omega_S']
        A_coeff = pipe['A_coeff']
        dim_bse = pipe['dim_bse']
        r_x_flat = pipe['r_x_flat']

        a_lat = model.a_lat
        b_lat = model.b_lat
        A_uc = 1 / (np.abs(1 / b_lat - 1 / a_lat))**2
        V_uc = A_uc * thickness * (N_top + N_bottom)

        d_x_all = A_coeff.conj().T @ r_x_flat
        d_y_all = A_coeff.conj().T @ pipe['r_y_flat']
        osc_x = np.abs(d_x_all)**2
        osc_y = np.abs(d_y_all)**2

        in_range = (Omega_S >= E_range[0]) & (Omega_S <= E_range[1])
        x_bright = np.where(in_range & (osc_x >= osc_y))[0]
        if len(x_bright) == 0:
            # Fallback: use strongest x-polarized state in the range.
            x_bright = np.where(in_range)[0]
            if len(x_bright) == 0:
                raise RuntimeError(f"No exciton found in E_range for N={N_top}/{N_bottom}")
            idx_sel = x_bright[np.argmax(osc_x[x_bright])]
        else:
            idx_sel = x_bright[np.argmin(Omega_S[x_bright])]

        # 5. Exciton electron/hole layer distributions -> dipole p_z
        A_S = A_coeff[:, idx_sel]
        A_3d = A_S.reshape(Nk, Nv, Nc)
        w_e_ck = np.sum(np.abs(A_3d)**2, axis=1)  # (Nk, Nc)
        w_h_vk = np.sum(np.abs(A_3d)**2, axis=2)  # (Nk, Nv)

        U_c = evecs[:, :, c_idx]  # (Nk, Nb, Nc)
        U_v = evecs[:, :, v_idx]  # (Nk, Nb, Nv)

        rho_e = np.zeros(Nb)
        rho_h = np.zeros(Nb)
        for ic in range(Nc):
            uc_sq = np.abs(U_c[:, :, ic])**2
            rho_e += np.sum(uc_sq * w_e_ck[:, ic:ic+1], axis=0)
        for iv in range(Nv):
            uv_sq = np.abs(U_v[:, :, iv])**2
            rho_h += np.sum(uv_sq * w_h_vk[:, iv:iv+1], axis=0)

        rho_e /= np.sum(rho_e)
        rho_h /= np.sum(rho_h)

        z_orb = np.zeros(Nb)
        z_orb[0:2] = +thickness * N_top / 2.0
        if Nb == 4:
            z_orb[2:4] = -thickness * N_bottom / 2.0

        z_e = float(np.dot(rho_e, z_orb))
        z_h = float(np.dot(rho_h, z_orb))
        dipole_z = z_e - z_h

        # zxx shift current from excitonic spectrum (many-body shift vector × |d|²)
        R_z_S0 = _many_body_z_shift_vector(
            A_coeff, pipe['z_eig'], v_idx, c_idx, Nk, Nv, Nc
        )
        integrand_S = R_z_S0 * np.abs(d_x_all)**2

        omegas = np.linspace(E_range[0], E_range[1], n_E)
        diff = omegas[:, None] - Omega_S[None, :]
        lorentz = (1.0 / np.pi) * eta / (diff**2 + eta**2)
        sigma_zxx = (lorentz @ integrand_S) / Nk

        e_charge = 1.602176634e-19
        hbar = 1.054571817e-34
        prefactor = (2 * np.pi * e_charge**2) / (hbar * V_uc) * 1E6
        sigma_zxx *= prefactor

        i_pk = _first_peak_idx(omegas, sigma_zxx)
        peak_energy = float(omegas[i_pk])
        peak_value = float(sigma_zxx[i_pk])

        rho_e_top = float(np.sum(rho_e[:2]))
        rho_h_top = float(np.sum(rho_h[:2]))
        rho_e_bot = float(np.sum(rho_e[2:])) if Nb == 4 else 0.0
        rho_h_bot = float(np.sum(rho_h[2:])) if Nb == 4 else 0.0

        print(f"  Lowest x-bright exciton: E={Omega_S[idx_sel]:.4f} eV, |d_x|^2={osc_x[idx_sel]:.4e}")
        print(f"  Layer weights: e(top/bot)=({rho_e_top:.3f}/{rho_e_bot:.3f}), "
              f"h(top/bot)=({rho_h_top:.3f}/{rho_h_bot:.3f})")
        print(f"  Dipole p_z=<z_e>-<z_h> = {dipole_z:.4f} A")
        print(f"  First |sigma^{{zxx}}| peak: omega={peak_energy:.4f} eV, sigma={peak_value:.4f} uA/V^2")

        results.append({
            'N_top': int(N_top),
            'N_bottom': int(N_bottom),
            'exciton_energy': float(Omega_S[idx_sel]),
            'osc_x': float(osc_x[idx_sel]),
            'dipole_z': dipole_z,
            'dipole_z_abs': float(np.abs(dipole_z)),
            'rho_e_top': rho_e_top,
            'rho_e_bottom': rho_e_bot,
            'rho_h_top': rho_h_top,
            'rho_h_bottom': rho_h_bot,
            'first_peak_energy': peak_energy,
            'first_peak_sigma_zxx': peak_value,
            'first_peak_sigma_zxx_abs': float(np.abs(peak_value)),
        })

    # 7. Summary plots
    labels = [f"{d['N_top']}/{d['N_bottom']}" for d in results]
    x_idx = np.arange(len(results))
    dipoles = np.array([d['dipole_z_abs'] for d in results])
    peaks = np.array([d['first_peak_sigma_zxx_abs'] for d in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_idx, dipoles, 'o-', color='tab:red', lw=2, label=r'$|p_z|$')
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel('N_top/N_bottom')
    axes[0].set_ylabel(r'$|p_z|$ (A)')
    axes[0].set_title('Lowest x-bright Exciton Dipole')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_idx, peaks, 's--', color='tab:blue', lw=2, label=r'first $|\sigma^{zxx}|$ peak')
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('N_top/N_bottom')
    axes[1].set_ylabel(r'$|\sigma^{zxx}_{peak}|$ ($\mu$A/V$^2$)')
    axes[1].set_title('First zxx Peak Amplitude')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname_trend = f"EM_xexciton_dipole_and_shift_peak{save_prefix}.png"
    plt.savefig(fname_trend, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(dipoles, peaks, c=np.arange(len(results)), cmap='viridis', s=80)
    for i, lab in enumerate(labels):
        plt.text(dipoles[i], peaks[i], f" {lab}", fontsize=9, va='bottom', ha='left')
    plt.xlabel(r'$|p_z|$ (A)')
    plt.ylabel(r'$|\sigma^{zxx}_{peak}|$ ($\mu$A/V$^2$)')
    plt.title('Relation: Exciton Dipole vs Shift-Current Peak')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname_corr = f"EM_xexciton_dipole_vs_shift_peak{save_prefix}.png"
    plt.savefig(fname_corr, dpi=300)
    plt.close()

    print("\n" + "=" * 68)
    print("Summary (lowest x-bright exciton vs first zxx peak)")
    for d in results:
        print(f"  N={d['N_top']}/{d['N_bottom']}: "
              f"|p_z|={d['dipole_z_abs']:.4f} A, "
              f"|sigma_peak|={d['first_peak_sigma_zxx_abs']:.4f} uA/V^2, "
              f"peak@{d['first_peak_energy']:.4f} eV")
    print(f"Saved: {fname_trend}")
    print(f"Saved: {fname_corr}")
    _save_dat(
        f"EM_xexciton_dipole_vs_shift_peak{save_prefix}.dat",
        np.array([
            [
                d['N_top'], d['N_bottom'],
                d['exciton_energy'], d['osc_x'],
                d['dipole_z'], d['dipole_z_abs'],
                d['rho_e_top'], d['rho_e_bottom'],
                d['rho_h_top'], d['rho_h_bottom'],
                d['first_peak_energy'],
                d['first_peak_sigma_zxx'],
                d['first_peak_sigma_zxx_abs'],
            ]
            for d in results
        ], dtype=float),
        "N_top N_bottom exciton_energy_eV osc_x dipole_z_A abs_dipole_z_A "
        "rho_e_top rho_e_bottom rho_h_top rho_h_bottom first_peak_energy_eV "
        "first_peak_sigma_zxx_uA_per_V2 abs_first_peak_sigma_zxx_uA_per_V2"
    )

    return results


def test_lowest_exciton_binding_convergence(N_top=1, N_bottom=1, twist_angle=0.0,
                                            n_k_bse_list=(12, 16, 20, 24, 30),
                                            k_range=0.15,
                                            n_val=2, n_cond=2,
                                            kappa=2.5, r0=5.0,
                                            band_window=None,
                                            use_gpu="auto", gpu_dtype="complex64",
                                            gpu_full_eigh_max_dim=32000,
                                            save_prefix=""):
    """
    Test convergence of the lowest-energy exciton binding energy versus n_k_bse.

    For each BSE k-grid, this computes

        E_bind = min(E_c - E_v) - Omega_0

    where Omega_0 is the lowest BSE exciton energy.

    Parameters
    ----------
    n_k_bse_list : sequence of int
        BSE k-grid sizes per direction. Each value uses an n_k_bse x n_k_bse grid.

    Returns
    -------
    results : list of dict
        One entry per grid, containing n_k_bse, Nk, dim_bse, qp_gap,
        lowest_exciton_energy, binding_energy, and elapsed_time_s.
    """
    print("Lowest Exciton Binding Energy Convergence Test")
    print(f"  Layers: N_top={N_top}, N_bottom={N_bottom}, twist={twist_angle:.4f} rad")
    print(f"  k_range={k_range:.6f}  kappa={kappa}, r0={r0} A")
    print(f"  BSE active space: n_val={n_val}, n_cond={n_cond}")
    print(f"  n_k_bse list: {list(n_k_bse_list)}")

    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom, twist_angle=twist_angle)
    results = []

    for n_k in n_k_bse_list:
        n_k = int(n_k)
        print("\n" + "=" * 72)
        print(f"Running n_k_bse={n_k} ({n_k}x{n_k} grid)")
        t0 = time.time()

        pipe = _run_bse_pipeline(model, k_range, n_k, n_val, n_cond,
                                 thickness=None, kappa=kappa, r0=r0,
                                 band_window=band_window,
                                 use_gpu=use_gpu, gpu_dtype=gpu_dtype,
                                 gpu_full_eigh_max_dim=gpu_full_eigh_max_dim)

        elapsed = time.time() - t0
        qp_gap = float(np.min(pipe['dE']))
        lowest_exciton = float(pipe['Omega_S'][0])
        binding_energy = qp_gap - lowest_exciton

        row = {
            'n_k_bse': n_k,
            'Nk': int(pipe['Nk']),
            'dim_bse': int(pipe['dim_bse']),
            'qp_gap': qp_gap,
            'lowest_exciton_energy': lowest_exciton,
            'binding_energy': float(binding_energy),
            'elapsed_time_s': float(elapsed),
        }
        results.append(row)

        print(f"Result n_k_bse={n_k}:")
        print(f"  QP gap                 = {qp_gap:.6f} eV")
        print(f"  Lowest exciton energy  = {lowest_exciton:.6f} eV")
        print(f"  Binding energy         = {binding_energy:.6f} eV")
        print(f"  Elapsed time           = {elapsed:.1f} s")

    n_k_arr = np.array([r['n_k_bse'] for r in results], dtype=int)
    nk_arr = np.array([r['Nk'] for r in results], dtype=int)
    dim_arr = np.array([r['dim_bse'] for r in results], dtype=int)
    qp_gap_arr = np.array([r['qp_gap'] for r in results])
    exciton_arr = np.array([r['lowest_exciton_energy'] for r in results])
    binding_arr = np.array([r['binding_energy'] for r in results])
    elapsed_arr = np.array([r['elapsed_time_s'] for r in results])

    data = np.column_stack([
        n_k_arr, nk_arr, dim_arr, qp_gap_arr, exciton_arr, binding_arr, elapsed_arr
    ])
    data_fname = f"EM_lowest_exciton_binding_convergence{save_prefix}.dat"
    np.savetxt(data_fname, data,
               header=("n_k_bse Nk dim_bse qp_gap_eV lowest_exciton_energy_eV "
                       "binding_energy_eV elapsed_time_s"),
               fmt=["%d", "%d", "%d", "%.10f", "%.10f", "%.10f", "%.4f"])

    plt.figure(figsize=(6, 4))
    plt.plot(n_k_arr, binding_arr, 'o-', color='tab:purple', lw=2,
             label='Binding energy')
    plt.xlabel(r'$n_{k,\mathrm{BSE}}$')
    plt.ylabel('Lowest exciton binding energy (eV)')
    plt.title(f'Lowest Exciton Binding Energy Convergence\n'
              f'N={N_top}/{N_bottom}, grid=$n_k^2$, kappa={kappa}, r0={r0} A')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 0.040)
    plt.legend()
    plt.tight_layout()
    fig_fname = f"EM_lowest_exciton_binding_convergence{save_prefix}.png"
    plt.savefig(fig_fname, dpi=300)
    plt.close()

    print("\nConvergence summary:")
    print("  n_k_bse    Nk    dim_BSE    QP_gap(eV)    Omega_0(eV)    E_bind(eV)")
    for r in results:
        print(f"  {r['n_k_bse']:7d} {r['Nk']:5d} {r['dim_bse']:10d} "
              f"{r['qp_gap']:11.6f} {r['lowest_exciton_energy']:13.6f} "
              f"{r['binding_energy']:12.6f}")
    print(f"\nSaved data: {data_fname}")
    print(f"Saved figure: {fig_fname}")

    return results


def plot_exciton_level(N_top=1, N_bottom=[2,7], twist_angle=0.0,
                                      E_range=(0.0, 1.0),
                                      k_range=0.15, n_k_bse=30,
                                      n_val=2, n_cond=2,
                                      kappa=2.5, r0=5.0,
                                      band_window=None,
                                      E_g=2.1, gamma_c = 0.58, gamma_v = -0.32,
                                      use_gpu="auto", gpu_dtype="complex64",
                                      gpu_full_eigh_max_dim=32000):
    bright_level = []
    for N_bot in range(N_bottom[0], N_bottom[1]+1):
        model = TwistedBPModel(N_top=N_top, N_bottom=N_bot, twist_angle=twist_angle)

        pipe = _run_bse_pipeline(model, k_range, n_k_bse, n_val, n_cond,
                                 thickness=None, kappa=kappa, r0=r0,
                                 band_window=band_window, use_gpu=use_gpu,
                                 gpu_dtype=gpu_dtype,
                                 gpu_full_eigh_max_dim=gpu_full_eigh_max_dim)

        Nk = pipe['Nk']
        Omega_S = pipe['Omega_S']
        A_coeff = pipe['A_coeff']
        r_b = pipe['r_b']
        dim_bse = pipe['dim_bse']

        # Compute oscillator strength |d^b_S|^2 for each exciton
        osc = {}
        for b_dir in ['x', 'y']:
            r_flat = r_b[b_dir].reshape(dim_bse)
            d_S = A_coeff.conj().T @ r_flat
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
    if N_top == 2:
        plt.plot([3,4,5,6],[0.63728, 0.55506, 0.49826, 0.45858],
                linestyle='None',marker='*', markersize=10, 
                markeredgewidth=1.0,
                markerfacecolor='white',markeredgecolor='lightcoral',
                label=r'X-bright (Exp.)')
        plt.plot([3,4,5,6],[0.80791, 0.80593, 0.80407, 0.78658],
                linestyle='None',marker='*', markersize=10, 
                markeredgewidth=1.0,
                markerfacecolor='white',markeredgecolor='royalblue',
                label=r'Y-bright (Exp.)')
    elif N_top == 3:
        plt.plot([3,4,5,6,7],[0.58383,0.49814,0.46267,0.41902,0.41307],
                linestyle='None',marker='*', markersize=10, 
                markeredgewidth=1.0,
                markerfacecolor='white',markeredgecolor='lightcoral',
                label=r'X-bright (Exp.)')
        plt.plot([3,4,5,6,7],[0.60913,0.58408,0.57478,0.5775,0.56734],
                linestyle='None',marker='*', markersize=10, 
                markeredgewidth=1.0,
                markerfacecolor='white',markeredgecolor='royalblue',
                label=r'Y-bright (Exp.)')
    elif N_top == 4:
        plt.plot([4,5,6,7],[0.50347, 0.47966, 0.41952, 0.39559],
                linestyle='None',marker='*', markersize=10, 
                markeredgewidth=1.0,
                markerfacecolor='white',markeredgecolor='lightcoral',
                label=r'X-bright (Exp.)')
        plt.plot([4,5,6,7],[0.51414, 0.51141, 0.47681, 0.47011],
                linestyle='None',marker='*', markersize=10, 
                markeredgewidth=1.0,
                markerfacecolor='white',markeredgecolor='royalblue',
                label=r'Y-bright (Exp.)')
    # line plot
    plt.plot(level_list, X_bright,
         linestyle='--', linewidth=1.0, color='red', alpha=0.6,
        #  label=r'X-bright (analytic)',
         zorder=0)
    plt.plot(level_list, Y_bright,
         linestyle='--', linewidth=1.0, color='blue', alpha=0.6,
        #  label=r'Y-bright (analytic)',
         zorder=0)
    
    # BSE results
    plt.plot(range(N_bottom[0], N_bottom[1]+1),bright_level[:, 1],
            linestyle='None',marker='v', markersize=8, 
            markerfacecolor='royalblue',
            markeredgewidth=0.4,
            markeredgecolor='black',
            label=r'Y-bright (Calc.)')
    plt.plot(range(N_bottom[0], N_bottom[1]+1), bright_level[:, 0],
            linestyle='None',marker='^', markersize=8, 
            markerfacecolor='lightcoral',
            markeredgewidth=0.4,
            markeredgecolor='black',
            label=r'X-bright (Calc.)')
    # plt.tight_layout()
    if N_top == 3:
        plt.ylim(0.0,1.0)
    elif N_top == 2:
        plt.ylim(0.0,1.2)
    elif N_top == 4:
        plt.ylim(0.2,0.8)
    plt.tick_params(which='both', direction='in', labelsize=11)
    plt.legend(frameon=False)
    plt.xlabel('Layer number (N)')
    plt.ylabel('Exciton Energy (eV)')
    fname = f"EM_exciton_level.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"\nSaved: {fname}")
    _save_dat(
        "EM_exciton_level_bse.dat",
        np.column_stack([
            np.arange(N_bottom[0], N_bottom[1] + 1),
            bright_level[:, 0],
            bright_level[:, 1],
        ]),
        "N_bottom X_bright_BSE_eV Y_bright_BSE_eV"
    )
    _save_dat(
        "EM_exciton_level_analytic.dat",
        np.column_stack([level_list, X_bright, Y_bright]),
        "N_bottom_continuous X_bright_analytic_eV Y_bright_analytic_eV"
    )


def lifetime():
    # =========================================================================
    # 1. Fundamental Physical Constants (SI Units)
    # =========================================================================
    epsilon_0 = 8.8541878128e-12  # Vacuum permittivity, F/m
    c = 299792458                 # Speed of light, m/s
    hbar = 1.054571817e-34        # Reduced Planck constant, J·s
    e = 1.602176634e-19           # Elementary charge, C
    m0 = 9.1093837015e-31         # Electron rest mass, kg
    kb = 1.380649e-23             # Boltzmann constant, J/K

    # =========================================================================
    # 2. Input Parameters from BSE Calculations
    # =========================================================================
    # Supercell area A: 143.7 Angstrom^2 -> Convert to m^2
    A = 143.7 * (1e-10) ** 2  
    
    # Exciton transition energy Es: 0.6208 eV -> Convert to Joules
    Es_eV = 0.6208
    Es = Es_eV * e            
    
    # Exciton total effective mass M: 0.50 m0 -> Convert to kg
    M = 0.48 * m0            
    
    # Temperature T: 300 K
    T = 300.0
    
    # Raw unnormalized square modulus of the transition dipole (Angstrom^2)
    raw_mu_squared = 68.74906 ** 2
    
    # Number of 2D k-points employed in the calculation
    Nk = 101 * 101
    
    # Normalize by the number of unit cells (Nk) according to the BSE definition
    mu_S_squared_ang = raw_mu_squared / Nk
    
    # Convert normalized transition dipole square from Angstrom^2 to m^2
    mu_S_squared = mu_S_squared_ang * (1e-10) ** 2

    # =========================================================================
    # 3. Calculation of Intrinsic Radiative Lifetime at 0 K: tau_S(0)
    # =========================================================================
    # Formula: tau_S(0) = (epsilon_0 * c * A * hbar^2) / (e^2 * mu_S^2 * Es)
    numerator_0K = epsilon_0 * c * A * (hbar ** 2)
    denominator_0K = (e ** 2) * mu_S_squared * Es
    
    tau_0K = numerator_0K / denominator_0K

    # =========================================================================
    # 4. Thermal Correction for Finite Temperature (300 K)
    # =========================================================================
    # Formula: Factor = (3 / 4) * (2 * M * c^2 / Es^2) * kb * T
    kbT = kb * T
    factor = 0.75 * ((2 * M * (c ** 2)) / (Es ** 2)) * kbT
    
    # Final radiative lifetime at finite temperature
    tau_300K = tau_0K * factor

    # =========================================================================
    # 5. Output Results and Intermediate Coefficients
    # =========================================================================
    print("--- Physical Parameters (SI Units) ---")
    print(f"Supercell Area A:               {A:.6e} m^2")
    print(f"Exciton Energy Es:              {Es:.6e} J ({Es_eV:.4f} eV)")
    print(f"Exciton Effective Mass M:       {M:.6e} kg ({M/m0:.3f} m0)")
    print(f"Normalized Dipole Square μ_s^2: {mu_S_squared:.6e} m^2")
    print(f"Thermal Energy k_B*T (300K):    {kbT:.6e} J")
    print("-" * 50)
    print("--- Intermediate and Final Lifetimes ---")
    print(f"0 K Numerator:                  {numerator_0K:.6e}")
    print(f"0 K Denominator:                {denominator_0K:.6e}")
    print(f"Radiative Lifetime tau_S(0):    {tau_0K:.6e} s ({tau_0K * 1e12:.3f} ps)")
    print(f"Thermal Scaling Factor:         {factor:.6e}")
    print(f"Radiative Lifetime tau_S(300):  {tau_300K:.6e} s ({tau_300K * 1e9:.3f} ns)")
    
    return tau_300K


def plot_current():
    import matplotlib.patches as mpatches
    # -------------------------
    # Data Calculation
    # -------------------------
    theta = np.linspace(0, 2 * np.pi, 1000)
    beta_zxx, beta_zyy = 14.0, -14.0
    
    I = beta_zxx * np.sin(theta)**2 + beta_zyy * np.cos(theta)**2
    I_abs = np.abs(I)

    # -------------------------
    # Initialize Polar Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

    # --- Seamless Filled Lobes with Edges ---
    # Positive (Red) Lobes
    ax.fill_between(theta, 0, I_abs, where=(I >= 0), 
                    color='red',linewidth=0.0, alpha=0.75, interpolate=True)
    # Negative (Blue) Lobes
    ax.fill_between(theta, 0, I_abs, where=(I < 0), 
                    color='blue',linewidth=0.0, alpha=0.75, interpolate=True)

    # -------------------------
    # Axes and Grid Styling
    # -------------------------
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.spines['polar'].set_linewidth(1.3)
    ax.grid(True, linestyle='--', linewidth=0.75, color='black', alpha=0.35)

    # Angular ticks (Every 45 degrees)
    angles = np.arange(0, 360, 45)
    ax.set_thetagrids(angles, labels=[str(a) for a in angles], fontsize=12)

    # Radial ticks & Labels on the left (180 degrees side)
    ax.set_rlim(0, 16)
    ax.set_rticks([5, 10, 15])
    ax.set_rlabel_position(180)  # Move the radial labels to the 180° line (left side)
    ax.tick_params(axis='y', labelsize=11)

    # Add the "Current (nA)" label on the far left side
    ax.text(np.deg2rad(189), 20, 'Current (nA)', fontsize=14, rotation=90, ha='center', va='center')

    # -------------------------
    # Legend and Formula
    # -------------------------
    red_patch = mpatches.Patch(color='red', alpha=0.75, label='Positive')
    blue_patch = mpatches.Patch(color='blue', alpha=0.75, label='Negative')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right', bbox_to_anchor=(1.15, 1.05), frameon=False, fontsize=13)

    plt.figtext(0.50, 0.04, r'$I = \beta_{zxx}\sin^2\theta + \beta_{zyy}\cos^2\theta$', ha='center', fontsize=18)

    # -------------------------
    # Save Image
    # -------------------------
    plt.savefig("EM_current_theta.png", dpi=400, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    # BP parameters
    # bulk from MP
    b_lat = 4.544
    a_lat = 3.295

    G_moire = 2 * np.pi * np.abs(1/b_lat - 1/a_lat)
    n_top = 3
    n_bottom = n_top
    twist_angle = np.pi / 2
    kappa=4.0
    r0=20.0
    # twist_angle = 0.0
    if n_top == 3:
        gamma_c = 0.58
        gamma_v = -0.32
        erange = (0.0, 1.00)
    elif n_top == 2:
        gamma_c = 0.49
        gamma_v = -0.42
        erange = (0.0, 1.20)
    elif n_top == 4:
        gamma_c = 0.57
        gamma_v = -0.32
        erange = (0.0, 0.80)
    eta = 0.020
    n_k_bse = 51
    n_cond = 1

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

    # # Projected Band structure
    # # # --------------------------------------------
    # plot_layer_projected_unfolded_bands(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                         k_fine_steps=360, y_lim=(-0.5, 1.0),
    #                                         lw=2.5, save_prefix="")

    # # # 3D Band Structure
    # # # --------------------------------------------
    # # plot_3d_bands(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    # #               k_range=G_moire/2, n_grid=60,
    # #               view_elev=15, view_azim=45)
    
    # # # Effective Mass
    # # # --------------------------------------------
    # calculate_effective_mass(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle)

    # # # Optical Conductivity
    # # # --------------------------------------------
    # calculate_optical_conductivity(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                n_k=240, n_E=500,eta=eta,
    #                                k_range=G_moire/2, E_range=erange)

    # # # # Optical Current
    # # # # --------------------------------------------
    # calculate_current(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                 n_k=240, n_E=500,eta=eta,shift_source="bse",
    #                                 n_k_bse=n_k_bse, n_val=2, n_cond=n_cond,
    #                                E_range=erange, k_range=G_moire/2,
    #                                kappa=kappa, r0=r0,)
    
    # # # Matrix Element Map (VBM -> CBM)
    # # # --------------------------------------------
    # plot_transition_matrix_elements(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                 band_indices=(1, 2),
    #                                 k_range=G_moire/2, n_k=160)
    
    # # # Bandgap scaling with N_bottom
    # # # --------------------------------------------
    # plot_bandgap_scaling(N_top=n_top, N_bottom=[n_top,10], twist_angle=twist_angle,
    #                         E_g=2.1, gamma_c = gamma_c, gamma_v = gamma_v,)


    # # # Shift Current Calculation
    # # # --------------------------------------------   
    # calculate_shift_current(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle, 
    #                         n_k=240, n_E=100,eta=eta,
    #                         E_range=erange, k_range=G_moire/2,)

    # # # Z-direction (out-of-plane) Shift Current
    # # # --------------------------------------------
    # calculate_z_shift_current(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                           n_k=240, n_E=250,eta=eta,
    #                         #   band_window=[0,1,2,2],
    #                           E_range=erange, k_range=G_moire/2)

    # # Lowest exciton binding energy convergence versus n_k_bse
    # # --------------------------------------------
    # test_lowest_exciton_binding_convergence(
    #     N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #     n_k_bse_list=[15, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    #     k_range=G_moire/2,
    #     n_val=2, n_cond=n_cond,
    #     kappa=kappa, r0=r0,
    #     use_gpu="auto", gpu_dtype="complex64",
    #     gpu_full_eigh_max_dim=32000,
    #     save_prefix=f"_N{n_top}_{n_bottom}"
    # )

    # # # BSE Excitonic Z-Shift Current
    # # # --------------------------------------------
    # calculate_bse_z_shift_current(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                n_k_bse=n_k_bse, n_val=2, n_cond=n_cond,
    #                                E_range=erange, k_range=G_moire/2,
    #                                kappa=kappa, r0=r0,eta=eta,
    #                                use_gpu="auto", gpu_dtype="complex64",
    #                                gpu_full_eigh_max_dim=32000,)
    
    # calculate_bse_inplane_shift_current(
    #     N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #     n_k_bse=n_k_bse, n_val=2, n_cond=n_cond,
    #     E_range=erange, k_range=G_moire/2,
    #     kappa=kappa, r0=r0, eta=eta,
    #     use_gpu="auto", gpu_dtype="complex128",
    # )
    
    # # # Exciton Oscillator Strength (stem plot)
    # # # --------------------------------------------
    # plot_exciton_oscillator_strength(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                   E_range=erange, eta=eta,
    #                                   k_range=G_moire/2, n_k_bse=n_k_bse,
    #                                   n_val=2, n_cond=n_cond,
    #                                   kappa=kappa, r0=r0,
    #                                   polarization='both')

    # # # BSE Excitonic Absorbance
    # # # --------------------------------------------
    # calculate_bse_absorbance(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                           E_range=erange, n_E=500, eta=eta,
    #                           k_range=G_moire/2, n_k_bse=n_k_bse,
    #                           n_val=2, n_cond=n_cond,
    #                           kappa=kappa, r0=r0,
    #                           plot_ipa_comparison=True,)
    
    # # # Excitonic Absorbance
    # # # --------------------------------------------
    # analyze_exciton_wavefunction(N_top=n_top, N_bottom=n_bottom, twist_angle=twist_angle,
    #                                 E_range=erange, eta=eta,
    #                                 k_range=G_moire/2, n_k_bse=n_k_bse,
    #                                 n_val=2, n_cond=n_cond,
    #                                 thickness=5.2,
    #                                 kappa=kappa, r0=r0,
    #                                 n_excitons=4,
    #                                 )

    # # # Excitonic Levels
    # # # --------------------------------------------
    # plot_exciton_level(N_top=n_top, N_bottom=[n_top,9], twist_angle=twist_angle,
    #                                   E_range=erange,
    #                                   k_range=G_moire/2, n_k_bse=n_k_bse,
    #                                   n_val=2, n_cond=n_cond,
    #                                   kappa=kappa, r0=r0,
    #                                   E_g=2.1, gamma_c = gamma_c, gamma_v = gamma_v,)

    # # Dipolar moment vs. Peak Shift
    # # --------------------------------------------
    # study_x_exciton_dipole_vs_shift_peak(
    #     layer_pairs=[(2, 2), (3, 3)],
    #     twist_angle=twist_angle,
    #     E_range=erange, n_E=500,eta=eta,
    #     k_range=G_moire/2, n_k_bse=n_k_bse,
    #     n_val=2, n_cond=n_cond,
    #     thickness=5.2,
    #     kappa=kappa, r0=r0,
    #     save_prefix="_N2N3"
    # )

    # # Exciton lifetime
    # # --------------------------------------------
    # lifetime()

    # plot_current()
