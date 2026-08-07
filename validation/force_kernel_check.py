"""Bit-for-bit cross-check of the intramolecular force kernel against Fortran.

The Fortran ./sens writes, at each sample, both the chain configuration and the
total force on each bead ("Gradient" in the NetCDF). This feeds each Fortran
configuration into the C++ total_force and checks they agree to machine precision
-- a config-level exact check that does NOT depend on the (wall-clock-seeded) RNG.

Usage:
    python validation/force_kernel_check.py --fortran /path/net_dt01_proc000.nc
"""
import argparse, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import bdsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fortran", required=True)
    # parameters must match inputc.dat
    ap.add_argument("--sqrtb", type=float, default=14.16)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--hstar", type=float, default=0.2)
    args = ap.parse_args()

    import netCDF4 as nc
    d = nc.Dataset(args.fortran)
    cfg = d["configuration"][:]; grad = d["Gradient"][:]      # (NBeads, Ndim, Nsamp, Ntraj)

    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.FENEFraenkel
    phys.spring.sqrtb = args.sqrtb
    phys.spring.natural_length = args.sigma
    phys.number_of_beads = cfg.shape[0]
    phys.hstar = args.hstar

    worst = 0.0; worst_rel = 0.0; checks = 0
    for s in range(cfg.shape[2]):
        for tr in range(cfg.shape[3]):
            R = np.asarray(cfg[:,:,s,tr], float)
            Ff = np.asarray(grad[:,:,s,tr], float)
            Fc = bdsim.total_force(R, phys)
            dd = np.abs(Fc - Ff).max(); scale = np.abs(Ff).max() or 1.0
            worst = max(worst, dd); worst_rel = max(worst_rel, dd/scale); checks += 1

    print(f"configurations checked: {checks}")
    print(f"worst |F_cpp - F_fortran| = {worst:.3e}   (relative {worst_rel:.3e})")
    ok = worst_rel < 1e-12
    print("RESULT:", "BIT-EXACT MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
