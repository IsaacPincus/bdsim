"""How the relaxation time scales with contour length under free-draining matching.

No simulation: everything here follows from the coarse-graining and the unit
conversion, so the whole sweep is a fraction of a second. It answers two
questions that decide whether a free-draining sweep is worth running at a given
L and Ns:

  1. What power of L does tau_1 (and hence lambda_eta,0) come out as?
  2. Over what range is that answer independent of Ns, which is the whole point
     of the matching?

The prediction. Matching the diffusivity fixes the total chain friction exactly,
zeta_tot = 6 pi eta_s R_H, with R_H taken from the DNA data. The free-draining
chain then relaxes as a Rouse chain with that friction,

    tau_1 = zeta_tot <R^2> / (3 pi^2 k_B T) = (2/pi) eta_s R_H <R^2> / (k_B T),

which contains no N_s: the spring constant H falls as N_s/L while the number of
beads rises, and the two cancel. Since <R^2> = 2 l_p L for an ideal wormlike
chain and R_H ~ L^nu_H,

    tau_1 ~ L^(1 + nu_H),   lambda_eta,0 = (pi^2/12) tau_1 ~ L^(1 + nu_H).

Both cancellations fail if H is capped, because then H no longer tracks N_s/L.

Run:  python validation/relaxation_scaling.py [--cap 0.05] [--plot out.png]
"""
import argparse, sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

LP_DNA, BP_NM = 147.0, 0.34          # persistence length in bp (= 50 nm), bp -> nm


def case(L_bp, Ns, cap, T, eta_s):
    """Everything the scaling argument needs, for one (L, Ns)."""
    from bdsim import coarse_grain as cg, dynamics as dyn
    p = cg.spring_parameters(L_bp, LP_DNA, Ns, bending="match_chain", max_H=cap)
    rh_nm = dyn.dna_hydrodynamic_radius_nm(L_bp, T, eta_s, length_unit_nm=BP_NM)
    hs, _ = dyn.free_draining_units(p, rh_nm, T, eta_s, BP_NM)
    u = dyn.physical_units(p, hs, T, eta_s, length_unit_nm=BP_NM,
                           hydrodynamic_radius_nm=rh_nm)
    tau1_s, _ = dyn.free_draining_relaxation_time(p, u)
    return dict(L=L_bp, Ns=Ns, tau1=tau1_s, RH_nm=rh_nm,
                R2_nm2=p.R2_chain_predicted * BP_NM ** 2,
                H=p.H, capped=cap is not None and p.H >= 0.999 * cap,
                ls_lp=p.ls / LP_DNA, kT=dyn.KB * T)


def slopes(x, y):
    return np.diff(np.log(y)) / np.diff(np.log(x))


def report(Ls, Nss, cap, T, eta_s):
    rows = {Ns: [case(L, Ns, cap, T, eta_s) for L in Ls] for Ns in Nss}
    kT = rows[Nss[0]][0]["kT"]

    print(f"{'L(kbp)':>8} {'ls/lp':>7} " +
          " ".join(f"{'tau1(Ns=%d)' % n:>11}" for n in Nss) +
          f" {'spread':>7} {'c':>6} {'capped':>7}")
    for i, L in enumerate(Ls):
        t = [rows[n][i]["tau1"] for n in Nss]
        r = rows[Nss[-1]][i]
        # tau_1 = c * eta_s * R_H * <R^2> / kT ; c -> 2/pi = 0.637 for a Rouse chain
        c = r["tau1"] * kT / (eta_s * r["RH_nm"] * 1e-9 * r["R2_nm2"] * 1e-18)
        print(f"{L/1000:8.3g} {r['ls_lp']:7.2f} " + " ".join(f"{x:11.4g}" for x in t) +
              f" {max(t)/min(t):6.2f}x {c:6.3f} {str(r['capped']):>7}")

    ref = rows[Nss[-1]]
    tau = np.array([r["tau1"] for r in ref])
    rh = np.array([r["RH_nm"] for r in ref])
    r2 = np.array([r["R2_nm2"] for r in ref])
    print(f"\n  measured  dln(tau_1)/dlnL        : " +
          " ".join(f"{x:5.2f}" for x in slopes(Ls, tau)))
    print(f"  predicted dln(R_H <R^2>)/dlnL    : " +
          " ".join(f"{x:5.2f}" for x in slopes(Ls, rh * r2)))
    print("  (c -> 2/pi = 0.637 and the two slope rows agree once ls/lp is large\n"
          "   enough that the discrete chain has reached its continuum limit.)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=None,
                    help="cap on H in pN/nm (the viscosity sweep uses 0.05); "
                         "omitted means no cap")
    ap.add_argument("--Ns", type=int, nargs="+", default=[20, 30, 40])
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--viscosity", type=float, default=1.0e-3)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    from bdsim import dynamics as dyn
    kT = dyn.KB * args.temperature
    cap = args.cap * BP_NM ** 2 / (kT * 1e21) if args.cap else None

    Ls = np.array([500, 1000, 2000, 5000, 10000, 20000, 48502,
                   100000, 200000, 500000], float)

    print(f"=== no cap on H ===")
    free = report(Ls, args.Ns, None, args.temperature, args.viscosity)
    if cap:
        print(f"\n=== H capped at {args.cap} pN/nm ===")
        capped = report(Ls, args.Ns, cap, args.temperature, args.viscosity)
        print("\nWhere the cap binds, H stops tracking N_s/L, so both cancellations\n"
              "fail at once: tau_1 becomes proportional to N_s and flattens to\n"
              "tau_1 ~ R_H ~ L^nu_H. Those points are not usable for scaling.")

    if args.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        for Ns in args.Ns:
            t = [r["tau1"] for r in free[Ns]]
            ax[0].plot(Ls / 1000, t, "o-", ms=4, label=f"$N_s$={Ns}")
            if cap:
                ax[0].plot(Ls / 1000, [r["tau1"] for r in capped[Ns]], "s--", ms=4,
                           alpha=0.6, label=f"$N_s$={Ns}, H capped")
            ax[1].plot(Ls[1:] / 1000, slopes(Ls, np.array(t)), "o-", ms=4,
                       label=f"$N_s$={Ns}")
        ref = free[args.Ns[-1]]
        pred = slopes(Ls, np.array([r["RH_nm"] * r["R2_nm2"] for r in ref]))
        ax[1].plot(Ls[1:] / 1000, pred, "k:", lw=2, label=r"$R_H\langle R^2\rangle$")
        ax[1].axhline(1.8, color="grey", ls="--", lw=1)
        ax[1].text(0.6, 1.82, "real DNA (excluded volume)", fontsize=7, color="grey")
        for a in ax:
            a.set_xscale("log"); a.set_xlabel("L (kbp)"); a.grid(alpha=0.3)
            a.legend(fontsize=7)
        ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\tau_1$ (s)")
        ax[0].set_title("free-draining relaxation time")
        ax[1].set_ylabel(r"$d\ln\tau_1/d\ln L$"); ax[1].set_ylim(0, 4)
        ax[1].set_title("local scaling exponent")
        fig.tight_layout(); fig.savefig(args.plot, dpi=120)
        print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    main()
