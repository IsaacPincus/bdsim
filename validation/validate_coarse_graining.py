"""Validate the WLC -> FENE-Fraenkel coarse-graining against its own targets.

Three checks, in increasing order of what they exercise:

  1. segment level -- the solved spring reproduces <R^2>_WLC(ls, lp) by
     construction; we confirm the solve converged and report the fourth moment,
     which is NOT fitted and so measures how good the shape is.
  2. chain level (Monte-Carlo) -- sample bond lengths and angles from the
     equilibrium distributions the parameters imply, assemble chains, and compare
     <R^2> with <R^2>_WLC(L, lp). This is the test the "saadat" bending constant
     fails and "match_chain" is designed to pass.
  4. figures (--plots) -- the end-to-end distribution and the force-extension
     curve, for the ideal wormlike chain and for the coarse-grained model. Both
     are rendered from Hamprecht-Kleinert fits (P(r) ~ r^(k+2)(1-r^b)^m) so the
     two are smooth analytic curves built the same way: the WLC fit uses its
     exact moments, the model fit uses its Monte-Carlo moments.

  3. chain level (Brownian dynamics) -- run the actual simulation with those
     parameters to equilibrium and measure <R^2>. This additionally checks that
     the force laws and the parameter mapping (sigma_H -> natural_length,
     dQ_H -> sqrtb, C -> stiffness) are consistent with the samplers.

Usage:  python validation/validate_coarse_graining.py [--bd]
"""
import sys, pathlib
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import bdsim
from bdsim import coarse_grain as cg

CASES = [(294, 10), (294, 30), (1223, 20), (7003, 30),
         (25000, 30), (25000, 80), (164000, 40)]
LP = 147.0          # DNA persistence length in base pairs


def mc_chain_R2(p, n_springs, n_chains=4000):
    """<R^2> (physical units) from chains sampled at the model's own equilibrium."""
    lH = p.dQ / p.dQ_H                       # Hookean length unit, in physical units
    R2 = np.empty(n_chains)
    for i in range(n_chains):
        R = bdsim.fene_fraenkel_bending_chain(n_springs + 1, p.sigma_H, p.dQ_H,
                                              p.C, seed=i)
        R2[i] = ((R[-1] - R[0]) ** 2).sum()
    return R2.mean() * lH ** 2, R2.std(ddof=1) / np.sqrt(n_chains) * lH ** 2


def bd_chain_R2(p, n_springs, n_traj=64, t_end=None, dt=None):
    """<R^2> from an equilibrium BD run with these parameters (Hookean units)."""
    phys = cg.to_phys_params(p)
    sim = bdsim.SimParams()
    # stiff springs need small steps; scale with the spring's curvature
    sim.dt = dt if dt else min(1e-3, 0.05 / max(1.0, p.dQ_H ** 2))
    sim.time_end = t_end if t_end else 200 * sim.dt * 20
    sim.implicit_loop_tol = 1e-5
    # Start from the model's own equilibrium: this asks whether the FORCES
    # preserve the distribution the samplers produce, not how fast a bad initial
    # guess relaxes.
    init = bdsim.Initial("fene_fraenkel_bending",
                         dict(sigma=p.sigma_H, dQ=p.dQ_H, stiffness=p.C))
    Rs = bdsim.run_ensemble(phys, sim, n_traj, bdsim.final_state, seed=1,
                            initial=init, backend="processes")
    lH = p.dQ / p.dQ_H
    m, e = bdsim.mean_stderr([bdsim.end_to_end_sq(R) for R in Rs])
    return m * lH ** 2, e * lH ** 2



def mc_reduced_moments(p, n_springs, n_chains=10000, orders=(1, 2, 3)):
    """<r^{2n}> of the coarse-grained chain, r = R/L, by Monte-Carlo sampling."""
    lH = p.dQ / p.dQ_H
    L = p.L
    R2 = np.empty(n_chains)
    for i in range(n_chains):
        R = bdsim.fene_fraenkel_bending_chain(n_springs + 1, p.sigma_H, p.dQ_H,
                                              p.C, seed=i)
        R2[i] = ((R[-1] - R[0]) ** 2).sum()
    r2 = R2 * lH ** 2 / L ** 2                    # (R/L)^2 per chain
    return [float(np.mean(r2 ** n)) for n in orders], np.sqrt(r2)


def make_plots(cases, bending="match_chain", path="validation/coarse_graining.png",
               n_chains=10000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 3.1 * len(cases)),
                             squeeze=False)
    r = np.linspace(1e-4, 1 - 1e-6, 800)

    for row, (L, Ns) in enumerate(cases):
        p = cg.spring_parameters(L, LP, Ns, bending=bending)

        wlc_mom = [cg.wlc_reduced_moment(L, LP, n) for n in (1, 2, 3)]
        wlc = cg.hk_fit(wlc_mom)
        mod_mom, r_samples = mc_reduced_moments(p, Ns, n_chains=n_chains)
        mod = cg.hk_fit(mod_mom)

        # --- distribution ---
        ax = axes[row][0]
        ax.hist(r_samples, bins=60, density=True, alpha=0.30, color="C1",
                label="model (Monte-Carlo)")
        if wlc:
            ax.plot(r, cg.hk_pdf(r, wlc), "k-", lw=2, label="WLC (exact moments)")
        if mod:
            ax.plot(r, cg.hk_pdf(r, mod), "C1--", lw=2, label="model (HK fit)")
        ax.set_xlabel("$R/L$"); ax.set_ylabel(r"$\psi(R/L)$")
        ax.set_title(f"$L={L}$ bp, $N_s={Ns}$, $L/l_p={L/LP:.2f}$", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # --- force-extension ---
        ax = axes[row][1]
        if wlc:
            ax.plot(r, cg.hk_force(r, wlc, L) * LP, "k-", lw=2,
                    label="WLC (exact moments)")
        if mod:
            ax.plot(r, cg.hk_force(r, mod, L) * LP, "C1--", lw=2, label="model")
        ax.plot(r, cg.marko_siggia_force(r, LP) * LP, ":", color="0.5", lw=1.5,
                label="Marko-Siggia")
        ax.set_xlabel("$R/L$"); ax.set_ylabel(r"$f\,l_p/k_\mathrm{B}T$")
        # log axis: below the most probable extension the force is compressive
        # (negative) and simply does not appear -- that is why the stiff case
        # only shows a curve above R/L ~ 0.8.
        ax.set_yscale("log"); ax.set_ylim(1e-2, 1e3)
        ax.set_title("force-extension (from moment fits -- see caveat)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


@dataclass
class Config:
    """Edit CONFIG below, or call run(Config(...)) from your own script."""

    bd: bool = False        # also run short Brownian-dynamics checks (slow)
    plots: bool = False     # write distribution + force-extension figures
    chains: int = 10000     # Monte-Carlo chains per case, for the figures


CONFIG = Config()


def run(cfg: Config = None):
    """Check the coarse-graining at segment level, chain level and (optionally) BD."""
    cfg = cfg or CONFIG

    print("1. SEGMENT LEVEL  (<Q^2> is fitted; <Q^4> is a free prediction)")
    print(f"{'L(bp)':>8} {'Ns':>4} {'N_K,s':>7} {'branch':>14} "
          f"{'<Q2>/<R2>-1':>12} {'<Q4>/<R4>-1':>12}")
    for L, Ns in CASES:
        p = cg.spring_parameters(L, LP, Ns)
        print(f"{L:8d} {Ns:4d} {p.N_ks:7.3f} {p.branch:>14} "
              f"{p.Q2_segment/p.R2_segment_target-1:+12.2e} "
              f"{p.Q4_segment/p.R4_segment_target-1:+12.2e}")

    print("\n2. CHAIN LEVEL, Monte-Carlo   <R^2>_model / <R^2>_WLC(L, lp)")
    print(f"{'L(bp)':>8} {'Ns':>4} | {'saadat':>18} | {'match_chain':>18}")
    for L, Ns in CASES:
        a = cg.spring_parameters(L, LP, Ns, bending="saadat")
        b = cg.spring_parameters(L, LP, Ns, bending="match_chain")
        ma, ea = mc_chain_R2(a, Ns)
        mb, eb = mc_chain_R2(b, Ns)
        print(f"{L:8d} {Ns:4d} | {ma/a.R2_chain_target:9.4f} +/- {ea/a.R2_chain_target:6.4f}"
              f" | {mb/b.R2_chain_target:9.4f} +/- {eb/b.R2_chain_target:6.4f}")

    if cfg.bd:
        print("\n3. CHAIN LEVEL, Brownian dynamics (equilibrium, free-draining)")
        print(f"{'L(bp)':>8} {'Ns':>4} {'BD/WLC':>10} {'MC/WLC':>10}")
        for L, Ns in [(25000, 30), (7003, 30), (294, 10)]:
            p = cg.spring_parameters(L, LP, Ns, bending="match_chain")
            mb, eb = bd_chain_R2(p, Ns)
            mm, _ = mc_chain_R2(p, Ns, n_chains=2000)
            print(f"{L:8d} {Ns:4d} {mb/p.R2_chain_target:10.4f} "
                  f"{mm/p.R2_chain_target:10.4f}")


    if cfg.plots:
        print()
        make_plots([(294, 30), (7003, 30), (25000, 30)], n_chains=cfg.chains)


if __name__ == "__main__":
    run()
