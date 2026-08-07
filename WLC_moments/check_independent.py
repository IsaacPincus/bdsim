"""Independent check: brute-force matrix exponential of the rigid-rotator
Hamiltonian in the orthonormal Legendre basis (D=3). No recursion used."""
import numpy as np
from scipy.linalg import expm
import mpmath as mp

mp.mp.dps = 40
N = 60
Lval = mp.mpf(1)

# H0 = diag(l(l+1)/2); Z_{l,l-1} = l/sqrt(4l^2-1)
def f_of_lam(lam):
    H = mp.zeros(N, N)
    for l in range(N):
        H[l, l] = mp.mpf(l * (l + 1)) / 2
    for l in range(1, N):
        z = mp.mpf(l) / mp.sqrt(4 * l**2 - 1)
        H[l, l-1] += lam / 2 * z
        H[l-1, l] += lam / 2 * z
    E = mp.expm(-H * Lval)
    return E[0, 0]

# high-order numerical derivatives of f at lam=0
for n in [1, 2, 3]:
    d = mp.diff(f_of_lam, 0, 2*n)          # d^{2n} f / dlam^{2n}
    Rz2n = d * mp.mpf(2)**(2*n)
    ang = mp.gamma(mp.mpf(3)/2) * mp.gamma(n + mp.mpf(1)/2) / (mp.sqrt(mp.pi) * mp.gamma(n + mp.mpf(3)/2))
    print(f"<R^{2*n}> at L=1 :", mp.nstr(Rz2n / ang, 20))
