"""
Reconstruction of the Mathematica notebook 'pgm15' from
B. Hamprecht & H. Kleinert, cond-mat/0305226 (Phys. Rev. E 71, 031803).

Rayleigh-Schroedinger recursion for the rigid rotator in a uniform field
    H = -Delta/2 + (lambda/2) z          on S^{D-1},  z = cos(theta)
whose unperturbed eigenstates are Gegenbauer polynomials C_l^{D/2-1}(z) with
    e0(l) = l(l+D-2)/2 .
Because <l|z|l'> is band-diagonal (l' = l+-1) the perturbation series closes
with a finite number of terms at every order -> exact rational moments.

Units: kappa_bar = 1, so the persistence length is xi = 2/(D-1)
(xi = 1 in D = 3).  L is kept symbolic.
"""

import sympy as sp

L = sp.Symbol('L', positive=True)


# --------------------------------------------------------------------------
# truncated power series in lambda, stored as coefficient lists c[0..M]
# --------------------------------------------------------------------------
def smul(a, b, M):
    out = [sp.Integer(0)] * (M + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if i + j > M:
                break
            if bj != 0:
                out[i + j] += ai * bj
    return out


def sadd(a, b):
    return [x + y for x, y in zip(a, b)]


def sinv(a, M):
    """1/a for a[0] != 0"""
    out = [sp.Integer(0)] * (M + 1)
    out[0] = sp.Integer(1) / a[0]
    for n in range(1, M + 1):
        s = sum(a[k] * out[n - k] for k in range(1, n + 1))
        out[n] = sp.together(-s / a[0])
    return out


def sexp(a, M):
    """exp(a) for a[0] == 0"""
    out = [sp.Integer(0)] * (M + 1)
    out[0] = sp.Integer(1)
    for n in range(1, M + 1):
        s = sum(k * a[k] * out[n - k] for k in range(1, n + 1))
        out[n] = sp.expand(s / n)
    return out


# --------------------------------------------------------------------------
# matrix elements
# --------------------------------------------------------------------------
def e0(l, D):
    return sp.Rational(l * (l + D - 2), 2)


def Wm1(k, D):
    """W_{-1}^{(k)} = <k|z|k-1>^2 ; zero for k = 0"""
    if k == 0:
        return sp.Integer(0)
    return sp.Rational(1) * k * (k + D - 3) / sp.Rational((2 * k + D - 2) * (2 * k + D - 4))


def alpha_ratios(kmax, D):
    """alpha_l / alpha_0, from alpha_l/alpha_{l+1} = <l|z|l+1>"""
    a = [sp.Integer(1)]
    for l in range(kmax):
        r = sp.sqrt(sp.Rational((l + 1) * (l + D - 2), (2 * l + D) * (2 * l + D - 2)))
        a.append(sp.simplify(a[-1] / r))
    return a


# --------------------------------------------------------------------------
# perturbation expansion of one level l
# --------------------------------------------------------------------------
def level(l, D, M, kmax):
    """returns (eps[0..M], gamma[k][0..M])"""
    eps = [sp.Integer(0)] * (M + 1)
    eps[0] = e0(l, D)
    gam = [[sp.Integer(0)] * (M + 1) for _ in range(kmax + 2)]
    gam[l][0] = sp.Integer(1)

    for i in range(1, M + 1):
        # eq. (2.6):  eps_i = sum_{n=+-1} gamma_{l+n,i-1} W_n^{(l)}
        acc = sp.Integer(0)
        if l + 1 <= kmax:
            acc += gam[l + 1][i - 1] * 1              # W_{+1} = 1 by normalisation
        if l - 1 >= 0:
            acc += gam[l - 1][i - 1] * Wm1(l, D)
        eps[i] = sp.nsimplify(sp.simplify(acc))

        # eq. (2.7) for k != l
        for k in range(kmax + 1):
            if k == l:
                continue
            num = sum(eps[j] * gam[k][i - j] for j in range(1, i))
            if k + 1 <= kmax:
                num -= gam[k + 1][i - 1] * 1
            if k - 1 >= 0:
                num -= gam[k - 1][i - 1] * Wm1(k, D)
            gam[k][i] = sp.simplify(num / (e0(k, D) - e0(l, D)))
    return eps, gam


# --------------------------------------------------------------------------
# generating function  f(L;lambda) = < exp(-(lambda/2) R_z) >
# --------------------------------------------------------------------------
def generating_function(D, M):
    kmax = M + M + 2
    alpha = alpha_ratios(kmax + 2, D)
    f = [sp.Integer(0)] * (M + 1)

    for l in range(0, M + 1):
        eps, gam = level(l, D, M, kmax)

        # exp(-E^{(l)} L) = exp(-e0 L) * exp(-L * sum_{j>=1} eps_j lam^j)
        arg = [sp.Integer(0)] * (M + 1)
        for j in range(1, M + 1):
            arg[j] = -L * eps[j]
        decay = sexp(arg, M)
        pref = sp.exp(-e0(l, D) * L)

        proj = gam[0][:]                       # <0|phi^l> / alpha_0
        num = smul(proj, proj, M)

        den = [sp.Integer(0)] * (M + 1)
        for k in range(kmax + 1):
            gk = gam[k][:]
            if all(x == 0 for x in gk):
                continue
            den = sadd(den, [alpha[k] ** 2 * c for c in smul(gk, gk, M)])

        term = smul(smul(num, sinv(den, M), M), decay, M)
        f = sadd(f, [pref * c for c in term])

    return [sp.simplify(sp.expand(c)) for c in f]


def moments(D, nmax):
    """returns dict n -> <R^{2n}>(L)"""
    M = 2 * nmax
    f = generating_function(D, M)
    out = {}
    for n in range(1, nmax + 1):
        Rz2n = f[2 * n] * sp.factorial(2 * n)   # recursion uses V=<k|z|j>, i.e. lam_paper = 2*lam_here
        # <cos^{2n}> on S^{D-1}
        ang = sp.gamma(sp.Rational(D, 2)) * sp.gamma(n + sp.Rational(1, 2)) / (
            sp.sqrt(sp.pi) * sp.gamma(n + sp.Rational(D, 2)))
        out[n] = sp.simplify(Rz2n / ang)
    return out


if __name__ == '__main__':
    D = 3
    xi = sp.Rational(2, D - 1)           # = 1 for D = 3
    mom = moments(D, 3)

    ref2 = 2 * (xi * L - xi ** 2 * (1 - sp.exp(-L / xi)))
    ref4 = (sp.Rational(4 * (D + 2), D) * L ** 2 * xi ** 2
            - 8 * L * xi ** 3 * (sp.Rational(D ** 2 + 6 * D - 1, D ** 2)
                                 - sp.Rational(D - 7, D + 1) * sp.exp(-L / xi))
            + 4 * xi ** 4 * (sp.Rational(D ** 3 + 23 * D ** 2 - 7 * D + 1, D ** 3)
                             - 2 * sp.Rational((D + 5) ** 2, (D + 1) ** 2) * sp.exp(-L / xi)
                             + 2 * sp.Rational((D - 5) ** 5, D ** 3 * (D + 1) ** 2)
                             * sp.exp(-2 * D * L / ((D - 1) * xi))))

    print("<R^2>  =", sp.simplify(mom[1]))
    print("  paper:", sp.simplify(ref2))
    print("  diff :", sp.simplify(mom[1] - ref2))
    print()
    print("<R^4>  =", sp.simplify(sp.expand(mom[2])))
    print("  paper:", sp.simplify(sp.expand(ref4)))
    print("  diff :", sp.simplify(sp.expand(mom[2] - ref4)))
    print()
    print("<R^6>  =", sp.simplify(sp.expand(mom[3])))
