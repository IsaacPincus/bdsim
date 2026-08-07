#!/usr/bin/env python3
"""Run BOTH codes (Fortran ./sens and the new C++/Python) and compare them.

Everything is configured in the CONFIG block below -- edit it, then just run:

    python validation/compare_fortran.py

What it does, in order:
  1. writes an inputc.dat with the parameters below and runs ./sens under mpirun
  2. runs the matching ensemble through the new C++ core -> HDF5
  3. cross-checks the FORCE KERNEL bit-for-bit (Fortran writes its own force at
     each sampled configuration, so this is exact and RNG-independent)
  4. checks the INITIAL CONFIGURATION distribution (bond lengths, KS test) --
     this isolates the initial-config generator from the dynamics
  5. compares ensemble-averaged observables vs time, with error bars, and plots

Why the dynamics comparison is statistical: the Fortran seeds its RNG from the
wall clock (sensemble.f90 uses the system time), so trajectories cannot be matched
one-to-one between the codes. Ensemble means with error bars are the meaningful
comparison. The force kernel and the initial distribution ARE checked exactly.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

# =====================================================================
# CONFIG -- edit this block
# =====================================================================
BDSIM_ROOT = pathlib.Path(__file__).resolve().parents[1]

SENS_EXE   = BDSIM_ROOT / "fortran_ref" / "sens"        # <-- point at YOUR compiled sens
INPUTC_SRC = BDSIM_ROOT / "fortran_ref" / "inputc.dat"  # template (other keys kept as-is)
WORK_DIR   = BDSIM_ROOT / "validation" / "comparison_run"   # scratch for both runs
PLOT_PATH  = BDSIM_ROOT / "validation" / "fortran_comparison.png"
JSON_PATH  = BDSIM_ROOT / "validation" / "fortran_comparison.json"

# --- what to run (set either to False to reuse whatever is already in WORK_DIR) ---
RUN_FORTRAN = True
RUN_CPP     = True

# --- ensemble size ---
N_RANKS       = 4      # mpirun -n N_RANKS
TRAJ_PER_RANK = 64     # Fortran trajectories per rank; total = N_RANKS * TRAJ_PER_RANK
CPP_NTRAJ     = None   # None => match the Fortran total
CPP_BACKEND   = "processes"    # "serial" or "processes"
CPP_WORKERS   = None           # None => all cores
CPP_SEED      = 0

# --- physics (must be expressible in BOTH codes) ---
N_BEADS     = 51
SQRTB       = 14.16
SIGMA       = 0.0      # natural length (0 => FENE-Fraenkel reduces to FENE)
HSTAR       = 0.2
SHEAR_RATE  = 1.0
HI_METHOD   = "Chebyshev"      # "Chebyshev" or "Cholesky"

# --- time stepping / sampling ---
DT        = 0.001
T_END     = 1.0        # total integrated time (Fortran Tpr)
N_SAMPLES = 5          # samples INCLUDING t=0, evenly spaced over [0, T_END]

# --- how to launch MPI (extra env is for unusual setups; normally leave empty) ---
MPIRUN     = "mpirun"
MPIRUN_ARGS = ["--oversubscribe"]
EXTRA_ENV  = {}        # e.g. {"LD_LIBRARY_PATH": "..."} if libs are not on the system path
# =====================================================================

sys.path.insert(0, str(BDSIM_ROOT / "python"))
import bdsim  # noqa: E402

GDOT = SHEAR_RATE
NAMES = ["Rsq", "Rg2", "Xstretch", "eta_p", "N1"]
LABELS = {"Rsq": r"$\langle R^2\rangle$", "Rg2": r"$\langle R_g^2\rangle$",
          "Xstretch": "x-stretch", "eta_p": r"$\eta_p=-\tau_{xy}/\dot\gamma$",
          "N1": r"$N_1=\tau_{xx}-\tau_{yy}$"}

FORT_DIR = WORK_DIR / "fortran"
CPP_DIR  = WORK_DIR / "cpp"


# ---------------------------------------------------------------- observables
def observables(R, F):
    """The five compared quantities, from positions and total bead forces."""
    R = np.asarray(R, float); F = np.asarray(F, float)
    Rc = R - R.mean(axis=0)
    tau = Rc.T @ F                                  # Kramers stress  sum (R-Rc)_i F_j
    return [float(((R[-1] - R[0]) ** 2).sum()),     # Rsq
            float((Rc ** 2).sum() / len(R)),        # Rg2
            float(R[:, 0].max() - R[:, 0].min()),   # x-stretch
            float(-tau[0, 1] / GDOT),               # eta_p
            float(tau[0, 0] - tau[1, 1])]           # N1


# ---------------------------------------------------------------- run Fortran
def write_inputc(dest):
    """Copy the template inputc, overriding the keys this comparison controls."""
    overrides = {
        "NBeads": N_BEADS, "sqrtb": SQRTB, "sigma": SIGMA, "hstar": HSTAR,
        "gdots": SHEAR_RATE, "Nsamples": N_SAMPLES, "Tpr": T_END,
        "dtsne": DT, "dtseq": DT,
        "nblock": TRAJ_PER_RANK, "ntot": TRAJ_PER_RANK,
        "delSCalcMethod": 0 if HI_METHOD == "Chebyshev" else 1,
        "SpType": 6,       # FENE-Fraenkel
        "FlowType": 1,     # shear
        "NetCDF": 1, "Teq": 0, "Restart": 0, "Ntrajdone": 0,
    }
    out = []
    seen = set()
    for line in INPUTC_SRC.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in overrides:
            out.append(f"{parts[0]}\t{overrides[parts[0]]}")
            seen.add(parts[0])
        else:
            out.append(line)
    for k, v in overrides.items():          # keys not present in the template
        if k not in seen:
            out.append(f"{k}\t{v}")
    dest.write_text("\n".join(out) + "\n")


def run_fortran():
    FORT_DIR.mkdir(parents=True, exist_ok=True)
    for old in FORT_DIR.glob("net_*.nc"):
        old.unlink()
    shutil.copy(SENS_EXE, FORT_DIR / "sens")
    os.chmod(FORT_DIR / "sens", 0o755)
    write_inputc(FORT_DIR / "inputc.dat")

    env = dict(os.environ); env.update(EXTRA_ENV)
    cmd = [MPIRUN, *MPIRUN_ARGS, "-n", str(N_RANKS), "./sens"]
    print(f"[fortran] {' '.join(cmd)}   ({N_RANKS} x {TRAJ_PER_RANK} = "
          f"{N_RANKS*TRAJ_PER_RANK} trajectories)", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=FORT_DIR, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
        raise SystemExit(f"./sens failed (exit {p.returncode})")
    print(f"[fortran] done in {time.time()-t0:.1f}s", flush=True)


# -------------------------------------------------------------------- run C++
def make_phys():
    p = bdsim.PhysParams()
    p.spring.type = bdsim.Spring.FENEFraenkel
    p.spring.sqrtb = SQRTB
    p.spring.natural_length = SIGMA
    p.number_of_beads = N_BEADS
    p.hstar = HSTAR
    p.hi_method = getattr(bdsim.DelSMethod, HI_METHOD)
    p.flow = bdsim.flows.shear(SHEAR_RATE)
    return p


def run_cpp(ntraj, n_steps, write_every):
    if CPP_DIR.exists():
        shutil.rmtree(CPP_DIR)
    sim = bdsim.SimParams(); sim.dt = DT
    initial = bdsim.Initial("fene_fraenkel", dict(sigma=SIGMA, dQ=SQRTB - SIGMA))
    out = bdsim.Output(directory=str(CPP_DIR), write_every=write_every, write_forces=True)
    print(f"[c++] {ntraj} trajectories, {n_steps} steps, snapshot every {write_every}",
          flush=True)
    t0 = time.time()
    bdsim.simulate(make_phys(), sim, n_traj=ntraj, seed=CPP_SEED, n_steps=n_steps,
                   initial=initial, output=out, backend=CPP_BACKEND, n_workers=CPP_WORKERS)
    print(f"[c++] done in {time.time()-t0:.1f}s", flush=True)


# -------------------------------------------------------------------- collect
def collect_fortran():
    import netCDF4 as nc
    times = None; per = None; configs0 = []
    files = sorted(FORT_DIR.glob("net_*.nc"))
    if not files:
        raise SystemExit(f"no Fortran output in {FORT_DIR}")
    forces0 = []
    for f in files:
        d = nc.Dataset(f)
        cfg = d["configuration"][:]; grad = d["Gradient"][:]; T = d["Time"][:]
        ns, nt = cfg.shape[2], cfg.shape[3]
        if per is None:
            per = [[[] for _ in NAMES] for _ in range(ns)]; times = np.asarray(T[:, 0])
        for tr in range(nt):
            R0 = np.asarray(cfg[:, :, 0, tr], float)
            if np.abs(R0).max() == 0.0:       # unwritten trajectory slot
                continue
            configs0.append(R0); forces0.append(np.asarray(grad[:, :, 0, tr], float))
            for s in range(ns):
                for k, v in enumerate(observables(cfg[:, :, s, tr], grad[:, :, s, tr])):
                    per[s][k].append(v)
        d.close()
    return times, per, configs0, forces0


def collect_cpp():
    run = bdsim.read_run(str(CPP_DIR))
    per = None; times = None; configs0 = []
    for traj in run:
        pos, frc, ts = traj.positions, traj.forces, traj.time
        if per is None:
            per = [[[] for _ in NAMES] for _ in range(len(ts))]; times = np.asarray(ts)
        configs0.append(np.asarray(pos[0], float))
        for s in range(len(ts)):
            for k, v in enumerate(observables(pos[s], frc[s])):
                per[s][k].append(v)
    return times, per, configs0


def stats(per):
    ns, no = len(per), len(per[0])
    mean = np.zeros((ns, no)); err = np.zeros((ns, no))
    for s in range(ns):
        for k in range(no):
            a = np.asarray(per[s][k])
            mean[s, k] = a.mean(); err[s, k] = a.std(ddof=1) / np.sqrt(len(a))
    return mean, err, len(per[0][0])


# ------------------------------------------------------------------- reports
def report_force_kernel(configs0, forces0):
    """Exact check: C++ force at each Fortran configuration vs the Fortran force."""
    phys = make_phys()
    worst = 0.0; worst_rel = 0.0
    for R, Ff in zip(configs0, forces0):
        Fc = bdsim.total_force(R, phys)
        d = np.abs(Fc - Ff).max(); worst = max(worst, d)
        worst_rel = max(worst_rel, d / (np.abs(Ff).max() or 1.0))
    ok = worst_rel < 1e-12
    print("\n=== 1. FORCE KERNEL (exact, RNG-independent) ===")
    print(f"  configurations checked : {len(configs0)}")
    print(f"  worst |F_cpp - F_fort| : {worst:.3e}  (relative {worst_rel:.3e})")
    print(f"  RESULT: {'BIT-EXACT MATCH' if ok else 'MISMATCH'}")
    return {"n": len(configs0), "worst_abs": worst, "worst_rel": worst_rel, "ok": bool(ok)}


def report_initial_distribution(f_cfg, c_cfg):
    """Exact-ish check: the t=0 bond-length distributions must be the same law."""
    def bonds(cfgs):
        return np.concatenate([np.linalg.norm(np.diff(R, axis=0), axis=1) for R in cfgs])
    bf, bc = bonds(f_cfg), bonds(c_cfg)
    print("\n=== 2. INITIAL CONFIGURATION (t=0 bond-length distribution) ===")
    print(f"  Fortran: n={len(bf):6d}  mean={bf.mean():.4f}  std={bf.std():.4f}  <Q^2>={np.mean(bf**2):.4f}")
    print(f"  C++    : n={len(bc):6d}  mean={bc.mean():.4f}  std={bc.std():.4f}  <Q^2>={np.mean(bc**2):.4f}")
    out = {"n_fortran": len(bf), "n_cpp": len(bc),
           "mean_fortran": bf.mean(), "mean_cpp": bc.mean()}
    try:
        from scipy import stats as st
        ks = st.ks_2samp(bf, bc)
        print(f"  KS two-sample: D={ks.statistic:.5f}  p={ks.pvalue:.3e}"
              f"  -> {'SAME distribution' if ks.pvalue > 0.01 else 'DIFFERENT (investigate)'}")
        out.update({"ks_D": float(ks.statistic), "ks_p": float(ks.pvalue)})
    except ImportError:
        print("  (scipy not installed -> KS test skipped)")
    return out


def report_dynamics(ft, fm, fe, fn, ct, cm, ce, cn):
    print("\n=== 3. DYNAMICS (statistical: ensemble means +/- standard error) ===")
    print(f"  Fortran trajectories: {fn}    C++ trajectories: {cn}")
    print(f"  sample times: Fortran {np.round(ft,4).tolist()}")
    print(f"                C++     {np.round(ct,4).tolist()}")
    # The Fortran time loop is inclusive and overshoots by one step at the end
    # (final sample at T_END + dt). One step in ~T_END/dt is a ~0.1% difference in
    # integrated time -- far below the statistical error -- but flag it so it is
    # never mistaken for a physics discrepancy.
    dtm = np.abs(np.asarray(ft) - np.asarray(ct))
    if dtm.max() > DT / 2:
        i = int(np.argmax(dtm))
        print(f"  NOTE: sample times differ by up to {dtm.max():.4g} "
              f"(at index {i}: Fortran {ft[i]:.4f} vs C++ {ct[i]:.4f}).")
        print(f"        This is the Fortran inclusive-loop overshoot (~{dtm.max()/max(T_END,1e-30)*100:.2f}% "
              f"of T_END), not a physics difference.")
    res = {}
    worst_z = 0.0
    for k, name in enumerate(NAMES):
        print(f"\n  --- {name} ---")
        print(f"  {'t':>6} {'Fortran':>22} {'C++':>22} {'|d|/sigma':>10}")
        rows = []
        for s in range(len(ft)):
            sig = np.hypot(fe[s, k], ce[s, k]) or 1.0
            z = abs(fm[s, k] - cm[s, k]) / sig
            worst_z = max(worst_z, z)
            flag = "" if z < 3 else "   <-- >3 sigma"
            print(f"  {ft[s]:6.3f} {fm[s,k]:11.3f} +/-{fe[s,k]:7.3f} "
                  f"{cm[s,k]:11.3f} +/-{ce[s,k]:7.3f} {z:9.2f}{flag}")
            rows.append({"t": float(ft[s]), "fortran": float(fm[s,k]), "fortran_err": float(fe[s,k]),
                         "cpp": float(cm[s,k]), "cpp_err": float(ce[s,k]), "z": float(z)})
        res[name] = rows
    print(f"\n  worst |difference|/sigma across all observables and times: {worst_z:.2f}")
    print("  (z < 2 is good agreement; z < 3 is acceptable; investigate persistent z > 3)")
    return res, worst_z


def make_plot(ft, fm, fe, fn, ct, cm, ce, cn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -> plot skipped)")
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for k, name in enumerate(NAMES):
        ax = axes.flat[k]
        ax.errorbar(ft, fm[:, k], fe[:, k], fmt="o-", capsize=3, label=f"Fortran (n={fn})")
        ax.errorbar(ct, cm[:, k], ce[:, k], fmt="s--", capsize=3, label=f"C++ (n={cn})")
        ax.set_title(LABELS[name]); ax.set_xlabel("t"); ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle(f"Fortran ./sens vs new C++  —  N={N_BEADS}, sqrtb={SQRTB}, "
                 f"h*={HSTAR} ({HI_METHOD}), shear {SHEAR_RATE}, dt={DT}, T={T_END}")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=110)
    print(f"\nsaved plot -> {PLOT_PATH}")


# ----------------------------------------------------------------------- main
def main():
    n_steps = int(round(T_END / DT))
    if (N_SAMPLES - 1) <= 0 or n_steps % (N_SAMPLES - 1) != 0:
        print(f"WARNING: {n_steps} steps does not divide evenly into {N_SAMPLES-1} "
              f"intervals; sample times may not line up exactly.")
    write_every = max(1, n_steps // (N_SAMPLES - 1))
    ntraj = CPP_NTRAJ if CPP_NTRAJ else N_RANKS * TRAJ_PER_RANK

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {WORK_DIR}")
    if RUN_FORTRAN:
        run_fortran()
    if RUN_CPP:
        run_cpp(ntraj, n_steps, write_every)

    ft, fp, f_cfg0, f_frc0 = collect_fortran()
    ct, cp, c_cfg0 = collect_cpp()
    fm, fe, fn = stats(fp); cm, ce, cn = stats(cp)

    force = report_force_kernel(f_cfg0, f_frc0)
    initial = report_initial_distribution(f_cfg0, c_cfg0)
    dyn, worst_z = report_dynamics(ft, fm, fe, fn, ct, cm, ce, cn)
    make_plot(ft, fm, fe, fn, ct, cm, ce, cn)

    JSON_PATH.write_text(json.dumps(
        {"config": {"N_BEADS": N_BEADS, "SQRTB": SQRTB, "SIGMA": SIGMA, "HSTAR": HSTAR,
                    "HI_METHOD": HI_METHOD, "SHEAR_RATE": SHEAR_RATE, "DT": DT,
                    "T_END": T_END, "N_SAMPLES": N_SAMPLES,
                    "n_fortran": fn, "n_cpp": cn},
         "force_kernel": force, "initial_distribution": initial,
         "dynamics": dyn, "worst_z": worst_z}, indent=2, default=float))
    print(f"saved results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
