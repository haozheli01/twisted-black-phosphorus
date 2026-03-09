import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

class TwistedBPModel:
    def __init__(self, N_top=1, N_bottom=1, twist_angle=0.0, scale_factor=0.5,
                 monolayer=True, stacking_shift=None):
        """
        Initialize TwistedBPModel with configurable parameters.

        Parameters:
            stacking_shift : array-like of shape (2,), optional
                In-plane displacement vector (in Angstroms) between the top and bottom
                layers at the twist interface.  Physically, this arises from the
                relative sliding of the puckered BP layers upon twisting.
                Introduces a phase exp(i k·d) in the interlayer coupling that
                breaks the sublattice conjugation symmetry K H* K = H, which is
                required for a nonzero shift current.
                Default: [0, b_lat/4] (quarter-period along armchair direction).
        """
        # monolayer BP parameters
        self.b_lat=4.588
        self.a_lat=3.296
        self.a1 = 2.22
        self.a2 = 2.24
        self.alpha1 = 96.5 * np.pi / 180
        self.alpha2 = 101.9 * np.pi / 180
        self.beta = 72.0 * np.pi / 180
        self.N_top = N_top
        self.N_bottom = N_bottom
        self.twist_angle = twist_angle
        self.scale_factor = scale_factor
        # Stacking shift for interface coupling phase
        if stacking_shift is None:
            self.stacking_shift = np.array([0.0, self.b_lat / 2.0])
        else:
            self.stacking_shift = np.asarray(stacking_shift, dtype=float)

        # tight-binding parameters
        # intralayer (in eV)
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

        # interlayer (in eV)
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
        num_k = len(k_points)
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
        
        H0 = np.zeros((num_k, 2, 2), dtype=np.complex128)
        H2 = np.zeros((num_k, 2, 2), dtype=np.complex128)
        H3 = np.zeros((num_k, 2, 2), dtype=np.complex128)

        H0[:, 0, 0] = tAA
        H0[:, 0, 1] = tAB
        H0[:, 1, 0] = np.conj(tAB)
        H0[:, 1, 1] = tAA
        H2[:, 0, 0] = tAD
        H2[:, 0, 1] = tAC
        H2[:, 1, 0] = np.conj(tAC)
        H2[:, 1, 1] = tAD
        H3[:, 0, 0] = tADp
        H3[:, 0, 1] = tACp
        H3[:, 1, 0] = np.conj(tACp)
        H3[:, 1, 1] = tADp

        return H0, H2, H3

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

        # === Assemble 2x2 derivative matrices ===
        def pack(diag, off):
            M = np.zeros((num_k, 2, 2), dtype=np.complex128)
            M[:, 0, 0] = diag;  M[:, 1, 1] = diag
            M[:, 0, 1] = off;   M[:, 1, 0] = np.conj(off)
            return M

        dH0_x = pack(gtAA_x, gtAB_x);  dH0_y = pack(gtAA_y, gtAB_y)
        dH2_x = pack(gtAD_x, gtAC_x);  dH2_y = pack(gtAD_y, gtAC_y)
        dH3_x = pack(gtADp_x, gtACp_x);  dH3_y = pack(gtADp_y, gtACp_y)

        return dH0_x, dH0_y, dH2_x, dH2_y, dH3_x, dH3_y

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

        # === Assemble 2x2 matrices ===
        def pack(diag, off):
            M = np.zeros((num_k, 2, 2), dtype=np.complex128)
            M[:, 0, 0] = diag;  M[:, 1, 1] = diag
            M[:, 0, 1] = off;   M[:, 1, 0] = np.conj(off)
            return M

        d2H0_xx = pack(gtAA_xx, gtAB_xx);  d2H0_yy = pack(gtAA_yy, gtAB_yy);  d2H0_xy = pack(gtAA_xy, gtAB_xy)
        d2H2_xx = pack(gtAD_xx, gtAC_xx);  d2H2_yy = pack(gtAD_yy, gtAC_yy);  d2H2_xy = pack(gtAD_xy, gtAC_xy)
        d2H3_xx = pack(gtADp_xx, gtACp_xx);  d2H3_yy = pack(gtADp_yy, gtACp_yy);  d2H3_xy = pack(gtADp_xy, gtACp_xy)

        return (d2H0_xx, d2H0_yy, d2H0_xy,
                d2H2_xx, d2H2_yy, d2H2_xy,
                d2H3_xx, d2H3_yy, d2H3_xy)

    def get_hamiltonians(self, k_points):
        """
        Compute the Hamiltonian for given k-points.
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)

        # untwisted block
        H0_top, H2_top, H3_top = self.basic_block(k_points, twist_angle=0.0)
        ham_top = np.zeros((num_k, 4, 4), dtype=np.complex128)
        ham_top[:,:2,:2] = H0_top
        ham_top[:,:2,2:4] = H2_top
        ham_top[:,2:4,:2] = H2_top
        ham_top[:,2:4,2:4] = H0_top
        ham_c_top = np.zeros((num_k, 4, 4), dtype=np.complex128)
        ham_c_top[:,:2,2:4] = H3_top

        # twisted block
        H0_bot, H2_bot, H3_bot = self.basic_block(k_points, twist_angle=self.twist_angle)

        ham_bot = np.zeros((num_k, 4, 4), dtype=np.complex128)
        ham_bot[:,:2,:2] = H0_bot
        ham_bot[:,:2,2:4] = H2_bot
        ham_bot[:,2:4,:2] = H2_bot
        ham_bot[:,2:4,2:4] = H0_bot
        ham_c_bot = np.zeros((num_k, 4, 4), dtype=np.complex128)
        ham_c_bot[:,:2,2:4] = H3_bot

        # num of layers in top and bottom
        N_tot = self.N_top + self.N_bottom
        ham = np.zeros((num_k, 4*N_tot, 4*N_tot), dtype=np.complex128)

        # top part
        for i in range(self.N_top):
            ham[:, 4*i:4*(i+1), 4*i:4*(i+1)] = ham_top
        if self.N_top > 1:
            for i in range(self.N_top-1):
                ham[:, 4*i:4*(i+1), 4*(i+1):4*(i+2)] = ham_c_top
                ham[:, 4*(i+1):4*(i+2), 4*i:4*(i+1)] = ham_c_top.conj().swapaxes(1,2)
        # bottom part
        for i in range(self.N_top, self.N_top + self.N_bottom):
            ham[:, 4*i:4*(i+1), 4*i:4*(i+1)] = ham_bot
        if self.N_bottom > 1:
            for i in range(self.N_top, self.N_top + self.N_bottom-1):
                ham[:, 4*i:4*(i+1), 4*(i+1):4*(i+2)] = ham_c_bot
                ham[:, 4*(i+1):4*(i+2), 4*i:4*(i+1)] = ham_c_bot.conj().swapaxes(1,2)
        # interface part — add stacking-shift phase to break conjugation symmetry
        phase = np.exp(1j * (k_points @ self.stacking_shift))[:, None, None]  # (Nk,1,1)
        interface = self.scale_factor * (ham_c_top + ham_c_bot) * phase
        ham[:, 4*(self.N_top-1):4*self.N_top, 4*self.N_top:4*(self.N_top+1)] = interface
        ham[:, 4*self.N_top:4*(self.N_top+1), 4*(self.N_top-1):4*self.N_top] = interface.conj().swapaxes(1,2)

        return ham

    def _assemble_block(self, block_top, block_bot, coupling_top, coupling_bot, num_k):
        """Helper to assemble full (Nk, dim_H, dim_H) matrix from 4x4 blocks."""
        N_tot = self.N_top + self.N_bottom
        M = np.zeros((num_k, 4 * N_tot, 4 * N_tot), dtype=np.complex128)
        for i in range(self.N_top):
            M[:, 4*i:4*(i+1), 4*i:4*(i+1)] = block_top
        if self.N_top > 1:
            for i in range(self.N_top - 1):
                M[:, 4*i:4*(i+1), 4*(i+1):4*(i+2)] = coupling_top
                M[:, 4*(i+1):4*(i+2), 4*i:4*(i+1)] = coupling_top.conj().swapaxes(1, 2)
        for i in range(self.N_top, N_tot):
            M[:, 4*i:4*(i+1), 4*i:4*(i+1)] = block_bot
        if self.N_bottom > 1:
            for i in range(self.N_top, N_tot - 1):
                M[:, 4*i:4*(i+1), 4*(i+1):4*(i+2)] = coupling_bot
                M[:, 4*(i+1):4*(i+2), 4*i:4*(i+1)] = coupling_bot.conj().swapaxes(1, 2)
        nt = self.N_top
        mixed = self.scale_factor * (coupling_top + coupling_bot)
        # Apply stacking-shift phase (or its derivative — caller supplies the right blocks)
        M[:, 4*(nt-1):4*nt, 4*nt:4*(nt+1)] = mixed
        M[:, 4*nt:4*(nt+1), 4*(nt-1):4*nt] = mixed.conj().swapaxes(1, 2)
        return M

    def _make_4x4_onsite(self, dH0, dH2, num_k):
        """Build 4x4 on-site block from 2x2 sub-blocks."""
        blk = np.zeros((num_k, 4, 4), dtype=np.complex128)
        blk[:, :2, :2] = dH0
        blk[:, :2, 2:4] = dH2
        blk[:, 2:4, :2] = dH2
        blk[:, 2:4, 2:4] = dH0
        return blk

    def _make_4x4_coupling(self, dH3, num_k):
        """Build 4x4 coupling block from 2x2 sub-block."""
        blk = np.zeros((num_k, 4, 4), dtype=np.complex128)
        blk[:, :2, 2:4] = dH3
        return blk

    def get_velocity_matrices(self, k_points):
        """
        Calculate velocity matrices vx, vy for the full Hamiltonian.
        v = dH/dk (units: eV * A)
        Returns vx, vy each (Nk, dim_H, dim_H).

        The interface coupling has the form: H_intf(k) * exp(i k·d),
        so d/dk_mu [H_intf * e^{ik·d}] = (dH_intf/dk_mu) * e^{ik·d}
                                          + i*d_mu * H_intf * e^{ik·d}
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)

        # Top layer (untwisted)
        dH0_x_t, dH0_y_t, dH2_x_t, dH2_y_t, dH3_x_t, dH3_y_t = \
            self.basic_block_velocity(k_points, twist_angle=0.0)
        # Bottom layer (twisted)
        dH0_x_b, dH0_y_b, dH2_x_b, dH2_y_b, dH3_x_b, dH3_y_b = \
            self.basic_block_velocity(k_points, twist_angle=self.twist_angle)

        # Also need H3 blocks for the phase-derivative term
        _, _, H3_top = self.basic_block(k_points, twist_angle=0.0)
        _, _, H3_bot = self.basic_block(k_points, twist_angle=self.twist_angle)

        # Stacking-shift phase
        d = self.stacking_shift
        phase = np.exp(1j * (k_points @ d))[:, None, None]  # (Nk,1,1)

        # Interface coupling H3 (before phase); H3_top/H3_bot are already (Nk,2,2)
        H3_intf = self.scale_factor * self._make_4x4_coupling(H3_top + H3_bot, num_k)

        # Assemble vx (non-interface part via _assemble_block without interface)
        vx = self._assemble_block(
            self._make_4x4_onsite(dH0_x_t, dH2_x_t, num_k),
            self._make_4x4_onsite(dH0_x_b, dH2_x_b, num_k),
            self._make_4x4_coupling(dH3_x_t, num_k),
            self._make_4x4_coupling(dH3_x_b, num_k), num_k)
        # Assemble vy
        vy = self._assemble_block(
            self._make_4x4_onsite(dH0_y_t, dH2_y_t, num_k),
            self._make_4x4_onsite(dH0_y_b, dH2_y_b, num_k),
            self._make_4x4_coupling(dH3_y_t, num_k),
            self._make_4x4_coupling(dH3_y_b, num_k), num_k)

        # Override the interface blocks in vx, vy with the correct derivative
        # d/dk_mu [H_intf * e^{ik·d}] = dH_intf/dk_mu * e^{ik·d} + i*d_mu * H_intf * e^{ik·d}
        nt = self.N_top
        dH3_x_intf = self.scale_factor * self._make_4x4_coupling(dH3_x_t + dH3_x_b, num_k)
        dH3_y_intf = self.scale_factor * self._make_4x4_coupling(dH3_y_t + dH3_y_b, num_k)

        vx_intf = (dH3_x_intf + 1j * d[0] * H3_intf) * phase
        vy_intf = (dH3_y_intf + 1j * d[1] * H3_intf) * phase

        vx[:, 4*(nt-1):4*nt, 4*nt:4*(nt+1)] = vx_intf
        vx[:, 4*nt:4*(nt+1), 4*(nt-1):4*nt] = vx_intf.conj().swapaxes(1, 2)
        vy[:, 4*(nt-1):4*nt, 4*nt:4*(nt+1)] = vy_intf
        vy[:, 4*nt:4*(nt+1), 4*(nt-1):4*nt] = vy_intf.conj().swapaxes(1, 2)

        return vx, vy

    def get_generalized_derivative_matrices(self, k_points):
        """
        w_munu = d^2H / dk_mu dk_nu for the full Hamiltonian.
        Returns w_xx, w_yy, w_xy each (Nk, dim_H, dim_H).

        For the interface coupling H_intf(k) * exp(ik·d):
        d^2/dk_mu dk_nu [H * e^{ik·d}] = d^2H/dk_mu dk_nu * e^{ik·d}
            + i*d_nu * dH/dk_mu * e^{ik·d} + i*d_mu * dH/dk_nu * e^{ik·d}
            - d_mu * d_nu * H * e^{ik·d}
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)

        # Top layer (untwisted)
        (d2H0_xx_t, d2H0_yy_t, d2H0_xy_t,
         d2H2_xx_t, d2H2_yy_t, d2H2_xy_t,
         d2H3_xx_t, d2H3_yy_t, d2H3_xy_t) = \
            self.basic_block_curvature(k_points, twist_angle=0.0)
        # Bottom layer (twisted)
        (d2H0_xx_b, d2H0_yy_b, d2H0_xy_b,
         d2H2_xx_b, d2H2_yy_b, d2H2_xy_b,
         d2H3_xx_b, d2H3_yy_b, d2H3_xy_b) = \
            self.basic_block_curvature(k_points, twist_angle=self.twist_angle)

        # Also need velocity and H3 blocks for phase-derivative cross terms
        _, _, dH2_x_t, dH2_y_t, dH3_x_t, dH3_y_t = \
            self.basic_block_velocity(k_points, twist_angle=0.0)
        _, _, dH2_x_b, dH2_y_b, dH3_x_b, dH3_y_b = \
            self.basic_block_velocity(k_points, twist_angle=self.twist_angle)
        _, _, H3_top = self.basic_block(k_points, twist_angle=0.0)
        _, _, H3_bot = self.basic_block(k_points, twist_angle=self.twist_angle)

        d = self.stacking_shift
        phase = np.exp(1j * (k_points @ d))[:, None, None]  # (Nk,1,1)

        # Interface blocks (before phase)
        H3_intf = self.scale_factor * self._make_4x4_coupling(
            H3_top + H3_bot, num_k)
        dH3_x_intf = self.scale_factor * self._make_4x4_coupling(dH3_x_t + dH3_x_b, num_k)
        dH3_y_intf = self.scale_factor * self._make_4x4_coupling(dH3_y_t + dH3_y_b, num_k)

        results = []
        nt = self.N_top
        d2H3_pairs = [
            (d2H3_xx_t, d2H3_xx_b, d[0], d[0]),  # xx
            (d2H3_yy_t, d2H3_yy_b, d[1], d[1]),  # yy
            (d2H3_xy_t, d2H3_xy_b, d[0], d[1]),  # xy
        ]
        dH3_mu_list = [dH3_x_intf, dH3_y_intf, dH3_x_intf]  # mu for each component
        dH3_nu_list = [dH3_x_intf, dH3_y_intf, dH3_y_intf]  # nu for each component

        for idx, ((d2H0_t, d2H2_t, d2H3_t, d2H0_b, d2H2_b, d2H3_b),
                   (d2H3_t_raw, d2H3_b_raw, d_mu, d_nu)) in enumerate(zip(
            [(d2H0_xx_t, d2H2_xx_t, d2H3_xx_t, d2H0_xx_b, d2H2_xx_b, d2H3_xx_b),
             (d2H0_yy_t, d2H2_yy_t, d2H3_yy_t, d2H0_yy_b, d2H2_yy_b, d2H3_yy_b),
             (d2H0_xy_t, d2H2_xy_t, d2H3_xy_t, d2H0_xy_b, d2H2_xy_b, d2H3_xy_b)],
            d2H3_pairs)):

            w = self._assemble_block(
                self._make_4x4_onsite(d2H0_t, d2H2_t, num_k),
                self._make_4x4_onsite(d2H0_b, d2H2_b, num_k),
                self._make_4x4_coupling(d2H3_t, num_k),
                self._make_4x4_coupling(d2H3_b, num_k), num_k)

            # Override interface blocks with full product-rule derivative
            d2H3_intf = self.scale_factor * self._make_4x4_coupling(d2H3_t_raw + d2H3_b_raw, num_k)
            w_intf = (d2H3_intf
                      + 1j * d_nu * dH3_mu_list[idx]
                      + 1j * d_mu * dH3_nu_list[idx]
                      - d_mu * d_nu * H3_intf) * phase
            w[:, 4*(nt-1):4*nt, 4*nt:4*(nt+1)] = w_intf
            w[:, 4*nt:4*(nt+1), 4*(nt-1):4*nt] = w_intf.conj().swapaxes(1, 2)
            results.append(w)

        return results[0], results[1], results[2]


# ============================================================
# Standalone analysis functions (following kp.py conventions)
# ============================================================

def cal_bands(N_top=4, N_bottom=4, twist_angle=0.0, scale_factor=0.5,
              n_points=100, y_lim=(-2, 1.5)):
    """Calculate and plot band structure along X -> Gamma -> Y."""
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom,
                           twist_angle=twist_angle, scale_factor=scale_factor)

    b_lat = model.b_lat
    a_lat = model.a_lat
    kx_max = np.pi / a_lat
    ky_max = np.pi / b_lat

    # Segment 1: X -> Gamma
    k1 = np.zeros((n_points, 2))
    k1[:, 0] = np.linspace(-kx_max, 0, n_points)
    # Segment 2: Gamma -> Y
    k2 = np.zeros((n_points, 2))
    k2[:, 1] = np.linspace(0, ky_max, n_points)

    k_path = np.vstack([k1, k2])
    dists = np.linalg.norm(np.diff(k_path, axis=0), axis=1)
    k_dist = np.concatenate([[0], np.cumsum(dists)])

    sym_pos = [0, k_dist[n_points - 1], k_dist[-1]]
    sym_labels = [r'$X$', r'$\Gamma$', r'$Y$']

    print(f"Calculating bands ({len(k_path)} k-points)...")
    H = model.get_hamiltonians(k_path)
    evals = np.linalg.eigvalsh(H)
    n = H.shape[-1]

    print(f"  dim_H = {n}")
    print('  Ref Band gap ≈ ',1.838-np.cos(np.pi/(int(n/4)+1))*2*0.712)
    print(f"  Band gap ≈ {np.min(evals[:, n//2] - evals[:, n//2 - 1]):.4f} eV")

    plot_2D_bands(k_dist, evals, sym_pos, sym_labels, y_lim)
    return model


def plot_2D_bands(k_dist, energies, sym_pos, sym_labels, y_lim):
    """Plot band structure."""
    plt.figure(figsize=(8, 8))
    plt.plot(k_dist, energies[:, 0], 'b-', lw=2.5, alpha=0.5, label='TB Bands')
    plt.plot(k_dist, energies[:, 1:], 'b-', lw=2.5, alpha=0.5)
    for pos in sym_pos:
        plt.axvline(pos, c='gray', ls='-', lw=0.5)
    plt.xticks(sym_pos, sym_labels)
    plt.ylim(y_lim)
    plt.xlim(k_dist[0], k_dist[-1])
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"TB_bands.png", dpi=200)
    plt.close()


def plot_3d_bands(N_top=4, N_bottom=4, twist_angle=0.0, scale_factor=0.5,
                  n_grid=40, bands_to_plot=4, k_range=0.15,
                  view_elev=30, view_azim=45):
    """Plot 3D band structure surface on a 2D k-grid."""
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom,
                           twist_angle=twist_angle, scale_factor=scale_factor)

    kx = np.linspace(-k_range, k_range, n_grid)
    ky = np.linspace(-k_range, k_range, n_grid)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])

    H_list = model.get_hamiltonians(k_points)
    evals = np.linalg.eigvalsh(H_list)
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
    ax.set_title('3D Band Structure (TB)')
    ax.set_box_aspect((1, 1, 1.5))
    ax.view_init(elev=view_elev, azim=view_azim)

    fname = f"TB_3D.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  Saved {fname}")


def calculate_optical_conductivity(N_top=4, N_bottom=4, twist_angle=0.0, scale_factor=0.5,
                                   E_range=(0.0, 2.0), n_E=500, eta=0.010, k_range=0.15,
                                   n_k=60):
    """
    Calculate interband optical absorption spectrum using velocity matrix elements.
    See kp.py docstring for the underlying Kubo-Greenwood formula.
    """
    print("Calculating optical conductivity spectrum (TB)...")
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom,
                           twist_angle=twist_angle, scale_factor=scale_factor)

    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])
    Nk = len(k_points)

    print(f"  Diagonalizing H for {Nk} k-points...")
    H_stack = model.get_hamiltonians(k_points)
    evals, evecs = np.linalg.eigh(H_stack)

    print("  Calculating velocity matrices...")
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

    print("  Summing transitions...")
    for i in range(mid_idx):
        for j in range(mid_idx, n_bands):
            delta_E = evals[:, j] - evals[:, i]
            M_ij_x = Mx2[:, i, j] / delta_E**2
            M_ij_y = My2[:, i, j] / delta_E**2

            diff = omegas[:, None] - delta_E[None, :]
            lorentz = (1 / np.pi) * eta / (diff**2 + eta**2)

            sigma_xx += np.sum(M_ij_x[None, :] * lorentz, axis=1)
            sigma_yy += np.sum(M_ij_y[None, :] * lorentz, axis=1)

    sigma_xx /= Nk
    sigma_yy /= Nk

    absorption_xx = sigma_xx * omegas
    absorption_yy = sigma_yy * omegas

    plt.figure(figsize=(8, 6))
    plt.plot(omegas, absorption_xx, 'r-', label=r'x-polarized', lw=2)
    plt.plot(omegas, absorption_yy, 'b--', label=r'y-polarized', lw=2)
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Optical Absorption (a.u.)')
    plt.title(f'TB Optical Absorption, $\\eta$={eta*1000:.1f} meV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)

    fname = f"TB_Absorption.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  Saved {fname}")


def plot_transition_matrix_elements(N_top=4, N_bottom=4, twist_angle=0.0, scale_factor=0.5,
                                    band_indices=None, n_k=60, k_range=0.15):
    """Plot |<j|v|i>|^2 contour maps in k-space."""
    print("Calculating transition matrix elements map (TB)...")
    model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom,
                           twist_angle=twist_angle, scale_factor=scale_factor)
    dim_H = 4 * (model.N_top + model.N_bottom)

    if band_indices is None:
        mid = dim_H // 2
        band_i = mid - 1
        band_j = mid
    else:
        band_i, band_j = band_indices
    print(f"  Mapping transition: Band {band_i} -> Band {band_j}")

    kx = np.linspace(-k_range, k_range, n_k)
    ky = np.linspace(-k_range, k_range, n_k)
    KX, KY = np.meshgrid(kx, ky)
    k_points = np.column_stack([KX.flatten(), KY.flatten()])

    H_stack = model.get_hamiltonians(k_points)
    _, evecs = np.linalg.eigh(H_stack)
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
    axes[0].set_title(r'$|\langle \psi_f | v_x | \psi_i \rangle|^2$ (x-pol)')
    axes[0].set_xlabel(r'$k_x$ ($\AA^{-1}$)');  axes[0].set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    axes[0].set_aspect('equal');  plt.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(KX, KY, Z_y, levels=40, cmap='plasma', vmin=0, vmax=vmax)
    axes[1].set_title(r'$|\langle \psi_f | v_y | \psi_i \rangle|^2$ (y-pol)')
    axes[1].set_xlabel(r'$k_x$ ($\AA^{-1}$)');  axes[1].set_ylabel(r'$k_y$ ($\AA^{-1}$)')
    axes[1].set_aspect('equal');  plt.colorbar(c2, ax=axes[1])

    plt.suptitle(f"TB Matrix Elements: Band {band_i} -> {band_j}")
    plt.tight_layout()

    fname = f"TB_M_B{band_i}-{band_j}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  Saved {fname}")


def calculate_shift_current(N_top=4, N_bottom=4, twist_angle=0.0, scale_factor=0.5,
                            E_range=(0.0, 4.0), n_E=400, eta=0.050, k_range=0.15,
                            n_k=60, band_window=None):
    r"""
    Shift current sigma^{abc}(omega) via gauge-invariant Sum-Over-States method.
    Ref: Phys. Rev. B 61, 5337 (2000)
    """
    comp_list=[('x', 'x', 'x'), ('x', 'y', 'y'), ('y', 'x', 'x'), ('y', 'y', 'y')]
    plt.figure(figsize=(8, 6))

    for comp in comp_list:
        a_dir, b_dir, c_dir = comp
        print(f"Calculating shift current sigma^{{{a_dir}{b_dir}{c_dir}}}(omega) "
            f"(TB, n_k={n_k}, eta={eta*1000:.0f} meV)...")

        model = TwistedBPModel(N_top=N_top, N_bottom=N_bottom,
                            twist_angle=twist_angle, scale_factor=scale_factor)

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
        w_map = {'xx': to_eig(w_xx), 'yy': to_eig(w_yy),
                'xy': to_eig(w_xy), 'yx': to_eig(w_xy)}

        v_a = v_map[a_dir];  v_b = v_map[b_dir];  v_c = v_map[c_dir]
        w_ac = w_map[a_dir + c_dir]

        mid = Nb // 2
        if band_window is None:
            v_idx = list(range(0, mid))
            c_idx = list(range(mid, Nb))
        else:
            v_idx = list(range(band_window[0], band_window[1] + 1))
            c_idx = list(range(band_window[2], band_window[3] + 1))

        omegas = np.linspace(E_range[0], E_range[1], n_E)
        sigma = np.zeros_like(omegas)

        print(f"  Transitions: {len(v_idx)} val x {len(c_idx)} cond, Nk={Nk}")
        eps_denom = 1e-5

        for n in v_idx:
            for m in c_idx:
                w_mn = evals[:, m] - evals[:, n]
                nonzero = w_mn > eps_denom

                f_n = 1.0 if n < mid else 0.0
                f_m = 1.0 if m < mid else 0.0
                f_nm = f_n - f_m
                if f_nm == 0.0:
                    continue

                r_b_mn = np.zeros(Nk, dtype=np.complex128)
                r_b_mn[nonzero] = v_b[nonzero, m, n] / (1j * w_mn[nonzero])

                # Term A
                termA = np.zeros(Nk, dtype=np.complex128)
                delta_a = v_a[nonzero, n, n] - v_a[nonzero, m, m]
                delta_c = v_c[nonzero, n, n] - v_c[nonzero, m, m]
                termA[nonzero] = (v_c[nonzero, n, m] * delta_a
                                + v_a[nonzero, n, m] * delta_c) / (-w_mn[nonzero])

                # Term B
                w_np = evals[:, n, None] - evals
                w_pm = evals - evals[:, m, None]
                valid_p = (np.abs(w_np) > eps_denom) & (np.abs(w_pm) > eps_denom)
                valid_p[:, n] = False;  valid_p[:, m] = False
                valid_p &= nonzero[:, None]

                num1 = v_c[:, n, :] * v_a[:, :, m]
                num2 = v_a[:, n, :] * v_c[:, :, m]
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

        sigma /= Nk

        # Physical prefactor: sigma has units of Å² so far (from velocity = dH/dk in eV·Å).
        # Full formula: sigma^{abc} = (2*pi*e^2) / (hbar * A_uc) * I(omega)
        # with an extra 1/hbar absorbed into the Lorentzian (delta in 1/eV → 1/(eV·s^{-1}))
        e_charge = 1.602176634e-19   # C
        hbar = 1.054571817e-34       # J·s
        A_uc = model.a_lat * model.b_lat  # Å² (orthorhombic unit cell)
        prefactor = (2 * np.pi * e_charge**2) / (hbar * A_uc) * 1E6  # → μA·Å / V²
        sigma *= prefactor

        plt.plot(omegas, sigma, lw=2,
                label=fr'$\sigma^{{{a_dir}{b_dir}{c_dir}}}(\omega)$')

    plt.axhline(0, color='k', lw=0.5, ls='--')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel(r'Shift Conductivity ($\mu$A$\cdot$Å/V$^2$)')
    plt.title(f'TB Shift Current Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(E_range)

    fname = f"TB_SC.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  Saved {fname}")
    return omegas, sigma


if __name__ == "__main__":

    # BP parameters
    b_lat=4.588
    a_lat=3.296
    G_moire = 2 * np.pi * np.abs(1/b_lat - 1/a_lat) # Moiré G vector magnitude for real lattice

    # Band structure
    cal_bands(N_top=1, N_bottom=1, twist_angle=np.pi/2, scale_factor=0.05,
              n_points=100, y_lim=(-10, 10))

    # 3D band structure
    plot_3d_bands(N_top=1, N_bottom=1, twist_angle=np.pi/2, scale_factor=0.05,
                  n_grid=60, bands_to_plot=4, k_range=G_moire/2)

    # Optical conductivity
    calculate_optical_conductivity(N_top=1, N_bottom=1, twist_angle=np.pi/2,
                                   scale_factor=0.05, n_k=200, n_E=500,
                                   eta=0.050, E_range=(0.0, 4.0), k_range=G_moire/2)

    # Transition matrix elements
    plot_transition_matrix_elements(N_top=1, N_bottom=1, twist_angle=np.pi/2,
                                    scale_factor=0.05, n_k=120, k_range=G_moire/2)

    # Shift current
    calculate_shift_current(N_top=1, N_bottom=1, twist_angle=np.pi/2,
                            scale_factor=0.05, n_k=300, n_E=400,
                            eta=0.010, E_range=(0.0, 4.0), k_range=G_moire/2)