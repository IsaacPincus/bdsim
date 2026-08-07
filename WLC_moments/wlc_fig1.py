"""
Reproduce Fig. 1 of Hamprecht & Kleinert, cond-mat/0305226.

Same recursion as wlc_moments.py but evaluated numerically (mpmath) so that
moments up to <R^20> are cheap.  Then the 3-parameter ansatz

    P_L(R) ~ r^(k+2) (1 - r^beta)^m ,   r = R/L

is fitted by matching three exact moments, as prescribed in Sec. III.
"""
import mpmath as mp
import numpy as np

mp.mp.dps = 60


# ---------- truncated series helpers (coefficient lists) -------------------
def smul(a, b, M):
    out = [mp.mpf(0)] * (M + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j in range(0, M - i + 1):
            if b[j] != 0:
                out[i + j] += ai * b[j]
    return out


def sinv(a, M):
    out = [mp.mpf(0)] * (M + 1)
    out[0] = 1 / a[0]
    for n in range(1, M + 1):
        out[n] = -sum(a[k] * out[n - k] for k in range(1, n + 1)) / a[0]
    return out


def sexp(a, M):
    out = [mp.mpf(0)] * (M + 1)
    out[0] = mp.mpf(1)
    for n in range(1, M + 1):
        out[n] = sum(k * a[k] * out[n - k] for k in range(1, n + 1)) / n
    return out


# ---------- the recursion --------------------------------------------------
def e0(l, D):
    return mp.mpf(l * (l + D - 2)) / 2


def Wm1(k, D):
    if k == 0:
        return mp.mpf(0)
    return mp.mpf(k * (k + D - 3)) / mp.mpf((2 * k + D - 2) * (2 * k + D - 4))


def alpha2(kmax, D):
    """(alpha_l/alpha_0)^2"""
    a = [mp.mpf(1)]
    for l in range(kmax + 1):
        r2 = mp.mpf((l + 1) * (l + D - 2)) / mp.mpf((2 * l + D) * (2 * l + D - 2))
        a.append(a[-1] / r2)
    return a


def moments_numeric(D, Lval, nmax):
    M = 2 * nmax
    kmax = 2 * M + 2
    a2 = alpha2(kmax + 2, D)
    Lv = mp.mpf(Lval)
    f = [mp.mpf(0)] * (M + 1)

    for l in range(M + 1):
        eps = [mp.mpf(0)] * (M + 1)
        eps[0] = e0(l, D)
        gam = [[mp.mpf(0)] * (M + 1) for _ in range(kmax + 2)]
        gam[l][0] = mp.mpf(1)

        for i in range(1, M + 1):
            acc = mp.mpf(0)
            if l + 1 <= kmax:
                acc += gam[l + 1][i - 1]
            if l >= 1:
                acc += gam[l - 1][i - 1] * Wm1(l, D)
            eps[i] = acc
            for k in range(kmax + 1):
                if k == l:
                    continue
                num = sum(eps[j] * gam[k][i - j] for j in range(1, i))
                if k + 1 <= kmax:
                    num -= gam[k + 1][i - 1]
                if k >= 1:
                    num -= gam[k - 1][i - 1] * Wm1(k, D)
                gam[k][i] = num / (e0(k, D) - e0(l, D))

        arg = [mp.mpf(0)] * (M + 1)
        for j in range(1, M + 1):
            arg[j] = -Lv * eps[j]
        decay = sexp(arg, M)
        pref = mp.e ** (-e0(l, D) * Lv)
        if pref < mp.mpf(10) ** (-mp.mp.dps - 10):
            continue

        num = smul(gam[0], gam[0], M)
        den = [mp.mpf(0)] * (M + 1)
        for k in range(kmax + 1):
            if any(gam[k]):
                den = [d + a2[k] * c for d, c in zip(den, smul(gam[k], gam[k], M))]
        term = smul(smul(num, sinv(den, M), M), decay, M)
        f = [x + pref * y for x, y in zip(f, term)]

    out = {}
    for n in range(1, nmax + 1):
        Rz = f[2 * n] * mp.factorial(2 * n)
        ang = mp.gamma(mp.mpf(D) / 2) * mp.gamma(n + mp.mpf(1) / 2) / (
            mp.sqrt(mp.pi) * mp.gamma(n + mp.mpf(D) / 2))
        out[n] = Rz / ang
    return out


# ---------- Section III: fit k, beta, m ------------------------------------
def model_moment(n, k, beta, m):
    """<r^2n> for P ~ r^(k+2)(1-r^beta)^m, eq. (3.2)"""
    A = (3 + k + 2 * n) / beta
    B = (3 + k) / beta
    return mp.exp(mp.loggamma(A) + mp.loggamma(B + m + 1)
                  - mp.loggamma(B) - mp.loggamma(A + m + 1))


def fit_params(target, ns):
    """match <r^2n> exactly for the three n in ns (log-space, multi-start)"""
    from scipy.optimize import least_squares
    from scipy.special import gammaln
    import numpy as np

    tgt = np.array([float(mp.log(target[n])) for n in ns])
    nn = np.array(ns, dtype=float)

    def logmom(p):
        k, beta, m = p[0] - 3.0, np.exp(p[1]), np.exp(p[2])
        A = (3 + k + 2 * nn) / beta
        B = (3 + k) / beta
        return gammaln(A) + gammaln(B + m + 1) - gammaln(B) - gammaln(A + m + 1)

    def res(p):
        if p[0] <= 1e-9:
            return np.full(3, 1e6)
        return logmom(p) - tgt

    m0 = 1.5 / float(target[1])
    starts = [(3.0, np.log(2.0), np.log(max(m0, 0.5))),
              (3.0, np.log(2.0), np.log(max(m0 / 3, 0.3))),
              (3.5, np.log(3.0), np.log(5.0)), (4.0, np.log(6.0), np.log(3.0)),
              (6.0, np.log(10.0), np.log(2.0)), (2.5, np.log(1.5), np.log(1.0)),
              (10.0, np.log(20.0), np.log(1.0)), (3.0, np.log(1.0), np.log(0.5))]
    best = None
    for s in starts:
        try:
            r = least_squares(res, np.array(s), xtol=1e-15, ftol=1e-15,
                              gtol=1e-15, max_nfev=20000)
        except Exception:
            continue
        c = float(np.max(np.abs(r.fun)))
        if c < 1e-9 and (best is None or c < best[0]):
            best = (c, r.x)
    if best is None:
        return None
    p = best[1]
    return float(p[0] - 3.0), float(np.exp(p[1])), float(np.exp(p[2]))


def run(xi_over_L, D=3):
    """paper's xi/L; our units have xi = 2/(D-1) = 1, so L = 1/(xi/L)"""
    Lv = mp.mpf(1) / mp.mpf(xi_over_L)
    nmax = max(3, int(round(4 * xi_over_L)) + 3)
    mom = moments_numeric(D, Lv, nmax)
    r2n = {n: mom[n] / Lv ** (2 * n) for n in mom}        # <r^2n>, r = R/L
    # published prescription (PRE 71, 031803): start at n = 4*xi, take the
    # next two higher even moments as well
    n0 = max(1, int(round(4 * xi_over_L)))
    ns = [n0, n0 + 1, n0 + 2]
    p = fit_params(r2n, ns)
    return r2n, ns, p


if __name__ == '__main__':
    print(f"{'xi/L':>8} {'<r^2>':>14} {'moments used':>16}   k, beta, m")
    results = {}
    for xi in [1/400, 1/100, 1/30, 1/10, 1/5, 1/2, 1, 2]:
        r2n, ns, p = run(xi)
        results[xi] = p
        print(f"{xi:8.5f} {float(r2n[1]):14.8f} {str(ns):>16}   "
              f"{'FAILED' if p is None else '%.4f, %.4f, %.4f' % p}")

    # ---- figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.integrate import quad

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    r = np.linspace(1e-6, 1 - 1e-9, 4000)
    for xi, p in results.items():
        if p is None:
            continue
        k, beta, m = p
        y = r ** (k + 2) * np.maximum(1 - r ** beta, 0) ** m
        norm = quad(lambda t: t ** (k + 2) * max(1 - t ** beta, 0) ** m, 0, 1,
                    limit=400)[0]
        ax.plot(r, y / norm, lw=1.6, label=rf'$\xi/L={xi:g}$' if xi >= 0.1
                else rf'$\xi/L=1/{round(1/xi)}$')
    ax.set_xlabel('$r = R/L$')
    ax.set_ylabel('$P_L(r)$  (normalised)')
    ax.set_title('Reproduction of Fig. 1, Hamprecht & Kleinert (cond-mat/0305226)')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig('fig1_reproduction.png', dpi=150)
    print('\nwrote fig1_reproduction.png')
