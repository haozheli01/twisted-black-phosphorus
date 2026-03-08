import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

class TwistedBPModel:
    def __init__(self, N_top=1, N_bottom=1, twist_angle=0.0, scale_factor=0.5,
                 monolayer=True):
        """
        Initialize TwistedBPModel with configurable parameters.
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

    def get_velocity_matrices(self, k_points, twist_angle):
        """
        Calculate velocity matrices vx, vy at k_points.
        v = dH/dk (units: eV * A)
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)
        kx_tmp, ky_tmp = k_points[:, 0], k_points[:, 1]
        kx = kx_tmp * np.cos(twist_angle) - ky_tmp * np.sin(twist_angle)
        ky = kx_tmp * np.sin(twist_angle) + ky_tmp * np.cos(twist_angle)

    def get_generalized_derivative_matrices(self, k_points, twist_angle):
        """
        w_munu = d^2H / dk_mu dk_nu
        Returns operator matrices w_xx, w_yy, w_xy representing the curvature of the Hamiltonian.
        """
        k_points = np.atleast_2d(k_points)
        num_k = len(k_points)
        kx_tmp, ky_tmp = k_points[:, 0], k_points[:, 1]
        kx = kx_tmp * np.cos(twist_angle) - ky_tmp * np.sin(twist_angle)
        ky = kx_tmp * np.sin(twist_angle) + ky_tmp * np.cos(twist_angle)

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
        # ham_c_top[:,2:4,:2] = H3_top

        # # twisted block
        # H0_bot, H2_bot, H3_bot = self.basic_block(k_points, twist_angle=self.twist_angle)

        # ham_bot = np.zeros((num_k, 4, 4), dtype=np.complex128)
        # ham_bot[:,:2,:2] = H0_bot
        # ham_bot[:,:2,2:4] = H2_bot
        # ham_bot[:,2:4,:2] = H2_bot
        # ham_bot[:,2:4,2:4] = H0_bot
        # ham_c_bot = np.zeros((num_k, 4, 4), dtype=np.complex128)
        # ham_c_bot[:,:2,2:4] = H3_bot

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
        # for i in range(self.N_top, self.N_top + self.N_bottom):
        #     ham[:, 4*i:4*(i+1), 4*i:4*(i+1)] = ham_bot
        # if self.N_bottom > 1:
        #     for i in range(self.N_top, self.N_top + self.N_bottom-1):
        #         ham[:, 4*i:4*(i+1), 4*(i+1):4*(i+2)] = ham_c_bot
        #         ham[:, 4*(i+1):4*(i+2), 4*i:4*(i+1)] = ham_c_bot.conj().swapaxes(1,2)
        # # interface part
        # ham[:, 4*(self.N_top-1):4*self.N_top, 4*self.N_top:4*(self.N_top+1)] = self.scale_factor * (ham_c_top + ham_c_bot)
        # ham[:, 4*self.N_top:4*(self.N_top+1), 4*(self.N_top-1):4*self.N_top] = self.scale_factor * (ham_c_top + ham_c_bot).conj().swapaxes(1,2)

        return ham


# Path: X(-kx_max, 0) -> Gamma(0,0) -> Y(0, ky_max)
b_lat=4.588
a_lat=3.296
dg_x = 2 * np.pi / a_lat
dg_y = 2 * np.pi / b_lat
kx_max = dg_x/2
ky_max = dg_y/2
n_points=100

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
model = TwistedBPModel(N_top=4, N_bottom=0, twist_angle=0.0)
H = model.get_hamiltonians(k_path)
# H = model.basic_block(k_path, twist_angle=0.0)
# H = model.get_hamiltonians(np.array([0.0,0.0]))
evals = np.linalg.eigvalsh(H)
n = H.shape[-1]
print('ref',1.838-np.cos(np.pi/(int(n/4)+1))*2*0.712)
print(np.min(evals[:,int(n/2)]-evals[:,int(n/2)-1]))
print(evals[100,:])

# Plot
plt.figure(figsize=(6, 5))
plt.plot(k_dist, evals, 'b-', lw=2)

# Ticks
tick_pos = [0, k_dist[n_points-1], k_dist[-1]]
tick_lab = ['X', r'$\Gamma$', 'Y']
for t in tick_pos:
    plt.axvline(t, color='gray', lw=0.5)
    
plt.xticks(tick_pos, tick_lab)
plt.ylabel("Energy (eV)")
plt.grid(True, alpha=0.3)
plt.ylim(-2,1.5)
plt.tight_layout()
plt.savefig("test.png", dpi=200)
plt.close()













# monolayer BP parameters
b_lat=4.588
a_lat=3.296
a1 = 2.22
a2 = 2.24
alpha1 = 96.5 * np.pi / 180
alpha2 = 101.9 * np.pi / 180
beta = 72.0 * np.pi / 180

# tight-binding parameters
# intralayer (in eV)
t1=-1.486
t2=3.729
t3=-0.252
t4=-0.071
t5= 0.019 # should be +0.019
t6=0.186
t7=-0.063
t8=0.101
t9=-0.042
t10=0.073
# interlayer (in eV)
t1p=0.524
t2p=0.180
t3p=-0.123
t4p=-0.168
t5p=0.005

# Path: X(-kx_max, 0) -> Gamma(0,0) -> Y(0, ky_max)
dg_x = 2 * np.pi / a_lat
dg_y = 2 * np.pi / b_lat
kx_max = dg_x/2
ky_max = dg_y/2
n_points=100

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

def monolayer_ham(k):
    k_points = np.atleast_2d(k)
    num_k = len(k_points)
    kx, ky = k_points[:, 0], k_points[:, 1]

    # Intralayer terms
    tAA = 2 * t3 * np.cos(2 * a1 * np.sin(alpha1/2) * kx ) \
        + 2 * t7 * np.cos((2 * a1 * np.sin(alpha1/2) + 2 * alpha2 * np.cos(beta)) * ky) \
        + 4 * t10 * np.cos(2 * a1 * np.sin(alpha1/2) * kx) * np.cos((2 * a1 * np.sin(alpha1/2) + 2 * alpha2 * np.cos(beta)) * ky)
    
    tAB = 2 * t1 * np.cos(a1 * np.sin(alpha1/2) * kx ) * np.exp(-1j * (a1 * np.cos(alpha1/2) * ky)) \
        + 2 * t4 * np.cos(a1 * np.sin(alpha1/2) * kx ) * np.exp(1j * (a1 * np.cos(alpha1/2) + 2 * a2 * np.cos(beta)) * ky) \
        + 2 * t8 * np.cos(3 * a1 * np.sin(alpha1/2) * kx) * np.exp(-1j * (a1 * np.cos(alpha1/2) * ky))
    
    tAC = t2 * np.exp(1j * a2 * np.cos(beta) * ky) \
        + t6 * np.exp(-1j * (2 * a1 * np.cos(alpha1/2) + a2 * np.cos(beta)) * ky) \
        + 2 * t9 * np.cos(2 * a1 * np.sin(alpha1/2) * kx) * np.exp(-1j * (a2 * np.cos(beta) + 2 * a1 * np.cos(alpha1/2)) * ky)
    
    tAD = 4 * t5 * np.cos(a1 * np.sin(alpha1/2) * kx) * np.cos((a1 * np.cos(alpha1/2) + a2 * np.cos(beta)) * ky)

    # Interlayer terms
    tADp = (4 * t3p * np.cos(2 * a1 * np.sin(alpha1/2) * kx ) + 2* t2p) \
         * np.cos((a1 * np.sin(alpha1/2) + alpha2 * np.cos(beta)) * ky)
    
    tACp = (2 * t1p * np.exp(1j * a2 * np.cos(beta) * ky) + 2 * t4p * np.exp(-1j * (2 * a1 * np.sin(alpha1/2) + a2 * np.cos(beta)) * ky)) \
         * np.cos(2 * a1 * np.sin(alpha1/2) * kx)
    
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
    ham = H0 + H2

    # analytic for val and cond in monolayer case
    # val = tAA + tAD - np.abs(tAB + tAC)
    # cond = tAA + tAD + np.abs(tAB + tAC)
    return ham

def bilayer_ham(k):
    k_points = np.atleast_2d(k)
    num_k = len(k_points)
    kx, ky = k_points[:, 0], k_points[:, 1]

    # Intralayer terms
    tAA = 2 * t3 * np.cos(2 * a1 * np.sin(alpha1/2) * kx ) \
        + 2 * t7 * np.cos((2 * a1 * np.sin(alpha1/2) + 2 * alpha2 * np.cos(beta)) * ky) \
        + 4 * t10 * np.cos(2 * a1 * np.sin(alpha1/2) * kx) * np.cos((2 * a1 * np.sin(alpha1/2) + 2 * alpha2 * np.cos(beta)) * ky)
    
    tAB = 2 * t1 * np.cos(a1 * np.sin(alpha1/2) * kx ) * np.exp(-1j * (a1 * np.cos(alpha1/2) * ky)) \
        + 2 * t4 * np.cos(a1 * np.sin(alpha1/2) * kx ) * np.exp(1j * (a1 * np.cos(alpha1/2) + 2 * a2 * np.cos(beta)) * ky) \
        + 2 * t8 * np.cos(3 * a1 * np.sin(alpha1/2) * kx) * np.exp(-1j * (a1 * np.cos(alpha1/2) * ky))
    
    tAC = t2 * np.exp(1j * a2 * np.cos(beta) * ky) \
        + t6 * np.exp(-1j * (2 * a1 * np.cos(alpha1/2) + a2 * np.cos(beta)) * ky) \
        + 2 * t9 * np.cos(2 * a1 * np.sin(alpha1/2) * kx) * np.exp(-1j * (a2 * np.cos(beta) + 2 * a1 * np.cos(alpha1/2)) * ky)
    
    tAD = 4 * t5 * np.cos(a1 * np.sin(alpha1/2) * kx) * np.cos((a1 * np.cos(alpha1/2) + a2 * np.cos(beta)) * ky)

    # Interlayer terms
    tADp = (4 * t3p * np.cos(2 * a1 * np.sin(alpha1/2) * kx ) + 2* t2p) \
         * np.cos((a1 * np.sin(alpha1/2) + alpha2 * np.cos(beta)) * ky)
    
    tACp = (2 * t1p * np.exp(1j * a2 * np.cos(beta) * ky) + 2 * t4p * np.exp(-1j * (2 * a1 * np.sin(alpha1/2) + a2 * np.cos(beta)) * ky)) \
         * np.cos(2 * a1 * np.sin(alpha1/2) * kx)
    
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
    
    ham = np.zeros((num_k, 4, 4), dtype=np.complex128)
    ham[:, :2, :2] = H0 + H2 + H3/2
    ham[:, 2:, 2:] = H0 + H2 - H3/2

    return ham


# Solve
H = bilayer_ham(k_path)
evals = np.linalg.eigvalsh(H)
print(np.min(evals[:,2]-evals[:,1]))
print(evals[100,:])

# H = monolayer_ham(k_path)
# evals = np.linalg.eigvalsh(H)
# print(np.min(evals[:,1]-evals[:,0]))
# print(evals[100,:])

# Plot
plt.figure(figsize=(6, 5))
plt.plot(k_dist, evals, 'b-', lw=2)

# Ticks
tick_pos = [0, k_dist[n_points-1], k_dist[-1]]
tick_lab = ['X', r'$\Gamma$', 'Y']
for t in tick_pos:
    plt.axvline(t, color='gray', lw=0.5)
    
plt.xticks(tick_pos, tick_lab)
plt.ylabel("Energy (eV)")
plt.grid(True, alpha=0.3)
plt.ylim(-2,1.5)
plt.tight_layout()
plt.savefig("BL.png", dpi=200)
plt.close()