#include "integrator.hpp"

#include "config.hpp"
#include "mobility.hpp"

#include <cmath>
#include <cstdio>

namespace bdsim {

namespace {

// Reporting for a corrector that ran out of iterations.
//
// This is the point where a trajectory actually fails. The iterate is accepted
// anyway (so the caller decides what to do), but it is no longer a solution of
// the implicit step: bonds typically land clamped on the FENE bound, where the
// spring force is enormous, and the chain is thrown a long way in one step.
// Any NaN that appears later -- usually from the Cholesky, once two beads have
// been driven on top of each other -- starts here.
//
// The warning is throttled deliberately. Once a chain is wrecked, every
// subsequent step also fails to converge and runs the full iteration cap, so
// unthrottled printing both floods the log and makes the run ~100x slower.
int nonconvergence_count = 0;

void report_nonconvergence(int iters, dp increment, dp tol) {
    ++nonconvergence_count;
    const int n = nonconvergence_count;
    const bool show = n <= 3 || (n <= 1000 && n % 100 == 0) || n % 10000 == 0;
    if (!show) return;
    std::fprintf(stderr,
        "bdsim warning: implicit corrector did not converge after %d iterations "
        "(final increment %.3e, tolerance %.1e)%s\n",
        iters, increment, tol,
        n == 1 ? " -- the step was accepted anyway; this is where the "
                 "trajectory fails, anything later is a consequence" : "");
    if (n == 3)
        std::fprintf(stderr, "bdsim warning: further non-convergence reports "
                             "will be throttled\n");
    if (n > 3)
        std::fprintf(stderr, "               (%d occurrences so far)\n", n);
}

// Euclidean norm of a bead field (the bond-solve residual measure).
dp field_norm(const Vec3Field& v) {
    dp s = 0.0;
    for (const auto& x : v) s += norm2(x);
    return std::sqrt(s);
}

// Apply the flow gradient to every bead:  (K.R)_nu = K . R_nu.
Vec3Field flow_velocity(const Mat3& K, const Vec3Field& R) {
    Vec3Field out(R.size());
    for (size_t i = 0; i < R.size(); ++i) out[i] = matvec(K, R[i]);
    return out;
}

// One chain, one integration. Each stage returns its result as a field, so the
// code mirrors the SDE  dR = [K.R + 1/4 D.F] dt + 1/sqrt2 B.dW.
class ChainIntegrator {
public:
    ChainIntegrator(const PhysParams& phys, const SimParams& sim)
        : phys_(phys), sim_(sim) {}

    void run(Vec3Field& R, Rng& rng) {
        for (dp t = sim_.time_start; t <= sim_.time_end + sim_.dt / 2.0; t += sim_.dt) {
            if (sim_.update_center_of_mass) recenter(R);

            const ChainGeometry geom = chain_geometry(R);
            const Vec3Field Fs = spring_force(geom, phys_.spring());   // implicit part
            // Explicit part: conservative non-spring forces plus anything the user
            // applies externally. The external force is a function of time, so the
            // predictor uses t and the corrector t + dt.
            const Vec3Field Fe = phys_.non_spring_force({R, geom})
                               + phys_.external_force(t);

            const Diffusion diff = phys_.mobility().at(R);             // D_n, built once
            const Vec3Field dW = diff.brownian_step(rng, sim_.dt);

            const Vec3Field R_pred = predictor(R, Fs, Fe, dW, diff, t);
            const Vec3Field ups    = upsilon(R, R_pred, Fs, Fe, dW, diff, t);
            R = solve_connectors(R_pred, ups, diff);
        }
    }

private:
    void recenter(Vec3Field& R) const {
        Vec3 com{0, 0, 0};
        for (const auto& r : R) com = com + r;
        com = (1.0 / phys_.nbeads()) * com;
        for (auto& r : R) r = r - com;
    }

    // Euler predictor:  R~ = R + [K.R + 1/4 D.(F^S + F^E)] dt + 1/sqrt2 dW
    Vec3Field predictor(const Vec3Field& R, const Vec3Field& Fs, const Vec3Field& Fe,
                        const Vec3Field& dW, const Diffusion& diff, dp t) const {
        const dp dt = sim_.dt;
        const Vec3Field DF = diff.apply(Fs + Fe);
        return R + dt * (flow_velocity(phys_.kappa(t), R) + 0.25 * DF) + sqrt2inv_ * dW;
    }

    // Corrector target Upsilon: the whole corrector RHS except the implicit
    // spring force F^S_{n+1}.
    //   Ups = R + [ 1/2(K.R + K.R~) + 1/8 D.F^S + 1/8 D.(F^E + F~^E) ] dt
    //           + 1/sqrt2 dW
    Vec3Field upsilon(const Vec3Field& R, const Vec3Field& R_pred, const Vec3Field& Fs,
                      const Vec3Field& Fe, const Vec3Field& dW, const Diffusion& diff,
                      dp t) const {
        const dp dt = sim_.dt;
        const ChainGeometry geom_pred = chain_geometry(R_pred);
        const Vec3Field Fe_pred = phys_.non_spring_force({R_pred, geom_pred})
                                + phys_.external_force(t + dt);
        const Vec3Field flow = 0.5 * (flow_velocity(phys_.kappa(t), R) +
                                      flow_velocity(phys_.kappa(t + dt), R_pred));
        const Vec3Field DFs = diff.apply(Fs);
        const Vec3Field DFe = diff.apply(Fe + Fe_pred);
        return R + dt * (flow + 0.125 * DFs + 0.125 * DFe) + sqrt2inv_ * dW;
    }

    // Implicit spring solve. Iterates the bond vectors to the fixed point of
    //   (1 + dt/4 f(Q)) Q_mu = Gamma_mu,
    //   Gamma_mu = D_mu[Ups] + 1/8 D_mu[D.F^S] dt + 1/4 F^c_mu dt,
    // with D_mu[X] = X_{mu+1} - X_mu the bond-difference operator.
    Vec3Field solve_connectors(const Vec3Field& R_pred, const Vec3Field& ups,
                               const Diffusion& diff) const {
        const dp dt = sim_.dt;
        const int N = phys_.nbeads();
        const int nbond = N - 1;

        Vec3Field diff_ups(nbond);
        for (int mu = 0; mu < nbond; ++mu) diff_ups[mu] = ups[mu + 1] - ups[mu];

        Vec3Field bond(nbond);
        for (int mu = 0; mu < nbond; ++mu) bond[mu] = R_pred[mu + 1] - R_pred[mu];

        const Vec3 anchor = R_pred[0];
        Vec3Field R_corr(N), R_prev = R_pred;

        const int max_iter = 100 * N;
        dp increment = 0.0;
        int iter = 0;
        for (; iter <= max_iter; ++iter) {
            const Vec3Field Fc  = connector_forces(chain_geometry_from_bonds(bond), phys_.spring());
            const Vec3Field DFs = diff.apply(bead_forces_from_connectors(Fc));

            R_corr[0] = anchor;
            for (int mu = 0; mu < nbond; ++mu) {
                const Vec3 gamma = diff_ups[mu]
                    + (0.125 * dt) * (DFs[mu + 1] - DFs[mu])
                    + (0.25 * dt) * Fc[mu];
                bond[mu] = solve_connector(gamma, dt, phys_.spring());
                R_corr[mu + 1] = R_corr[mu] + bond[mu];
            }

            increment = field_norm(R_corr - R_prev) / N;
            if (increment < sim_.implicit_loop_tol) break;
            R_prev = R_corr;
        }
        // Falling out of the loop means the fixed point was never reached. The
        // iterate is returned regardless, but it is not a solution of the step.
        if (iter > max_iter)
            report_nonconvergence(max_iter, increment, sim_.implicit_loop_tol);
        return R_corr;
    }

    PhysicalModel phys_;         // owns all the physics, built from PhysParams
    const SimParams& sim_;
    const dp sqrt2inv_ = 1.0 / std::sqrt(2.0);
};

}  // namespace

void time_integrate_chain(Vec3Field& R, const PhysParams& phys, const SimParams& sim,
                          Rng& rng) {
    ChainIntegrator(phys, sim).run(R, rng);
}

} // namespace bdsim
