"""Physics validation: equilibrium chain statistics vs Rouse theory.

In the code's reduced units the equilibrium Hookean bond has <Q^2> = 3 (unit
variance per Cartesian component), so for an N-bead ideal chain:
    <R^2>  = 3 (N - 1)
    <Rg^2> = (3/6) (N^2 - 1) / N
Equilibrium averages are independent of hydrodynamic interaction, so free
draining and HI (Cholesky / Chebyshev) must agree within statistics.

Usage:  python validate_physics.py [n_traj]     (default 400; try 4000 for tight)
"""
import sys
import numpy as np
import sys, pathlib
# Run standalone from anywhere: put the bdsim package (and its compiled
# _bdsim extension) on the path without needing PYTHONPATH set.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import bdsim


def equilibrium(n, n_traj, seed=1, hstar=0.0, method=None):
    phys = bdsim.PhysParams()
    phys.spring.type = bdsim.Spring.Hook
    phys.spring.sqrtb = 1000.0
    phys.number_of_beads = n
    phys.hstar = hstar
    if method is not None:
        phys.hi_method = method
    sim = bdsim.SimParams()
    sim.time_end, sim.dt = 8.0, 0.05
    sim.implicit_loop_tol = 1e-4
    # gaussian_chain(bond_std=1) already samples the Hookean equilibrium.
    return bdsim.run_ensemble(phys, sim, n_traj, seed=seed, n_beads=n,
                              properties=("Rsq", "Rg_sq"), backend="processes")


def main():
    n_traj = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print(f"Equilibrium validation ({n_traj} trajectories/case)\n")
    ok = True

    print(f"{'N':>3}  {'<R^2>':>18}  {'theory':>8}   {'<Rg^2>':>18}  {'theory':>8}")
    for n in (5, 10, 20):
        res = equilibrium(n, n_traj)
        r2, r2e = res["Rsq"]; rg, rge = res["Rg_sq"]
        r2_th = 3.0 * (n - 1)
        rg_th = 0.5 * (n * n - 1) / n
        # pass if theory within 3 standard errors
        r2_ok = abs(r2 - r2_th) < 3 * r2e
        rg_ok = abs(rg - rg_th) < 3 * rge
        ok = ok and r2_ok and rg_ok
        print(f"{n:>3}  {r2:8.3f} +/- {r2e:5.3f}  {r2_th:8.2f} {'ok' if r2_ok else 'XX'}"
              f"   {rg:8.3f} +/- {rge:5.3f}  {rg_th:8.2f} {'ok' if rg_ok else 'XX'}")

    # HI must not change equilibrium: compare <R^2> for N=10 across methods.
    print("\nHI-invariance of equilibrium (<R^2>, N=10):")
    base = equilibrium(10, n_traj, seed=1)["Rsq"]
    chol = equilibrium(10, n_traj, seed=1, hstar=0.2, method=bdsim.DelSMethod.Cholesky)["Rsq"]
    cheb = equilibrium(10, n_traj, seed=1, hstar=0.2, method=bdsim.DelSMethod.Chebyshev)["Rsq"]
    print(f"  free-draining {base[0]:8.3f} +/- {base[1]:.3f}")
    print(f"  HI Cholesky   {chol[0]:8.3f} +/- {chol[1]:.3f}")
    print(f"  HI Chebyshev  {cheb[0]:8.3f} +/- {cheb[1]:.3f}")
    spread = max(base[0], chol[0], cheb[0]) - min(base[0], chol[0], cheb[0])
    tol = 3 * max(base[1], chol[1], cheb[1])
    inv_ok = spread < tol
    ok = ok and inv_ok
    print(f"  spread {spread:.3f}  (tol {tol:.3f}) -> {'ok' if inv_ok else 'XX'}")

    print("\n" + ("ALL PHYSICS CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
