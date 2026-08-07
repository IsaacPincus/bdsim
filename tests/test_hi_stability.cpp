// test_hi_stability.cpp — robustness of the semi-implicit step with hydrodynamic
// interaction. These guard the failure mode where the Chebyshev square root
// diverged on stretched chains (spectral bounds missing the smallest eigenvalue),
// producing an unbounded Brownian displacement that a bug in the cubic root solver
// then turned into an exploding bond. Three layers are checked:
//   (1) the implicit spring solve keeps the bond finitely extensible for ANY gama;
//   (2) the Chebyshev square root brackets the spectrum / stays a true sqrt even on
//       a strongly stretched configuration, and never returns a divergent B.X0;
//   (3) a full FENE-Fraenkel + HI shear run stays bounded over many steps and the
//       Chebyshev and Cholesky methods agree statistically.
#include "../src/hydrodynamics.hpp"
#include "../src/integrator.hpp"
#include "../src/linalg.hpp"
#include "../src/spring.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace bdsim;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%-52s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

// Build a stretched chain: bonds near full extension along x with small jitter.
static Vec3Field stretched_chain(int N, dp bond) {
    Vec3Field R(N);
    for (int i = 0; i < N; ++i)
        R[i] = {bond * i, 0.01 * std::sin(1.3 * i), 0.01 * std::cos(0.7 * i)};
    return R;
}


// Exact extreme eigenvalues of a small symmetric matrix by cyclic Jacobi.
// Only used by the test, so clarity beats speed.
static std::pair<double,double> jacobi_extremes(Matrix A) {
    const int n = A.n;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) off += A(i,j)*A(i,j);
        if (off < 1e-18) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                if (std::fabs(A(p,q)) < 1e-300) continue;
                const double theta = (A(q,q) - A(p,p)) / (2.0 * A(p,q));
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta*theta + 1.0));
                const double c = 1.0 / std::sqrt(t*t + 1.0), sn = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = A(k,p), akq = A(k,q);
                    A(k,p) = c*akp - sn*akq;
                    A(k,q) = sn*akp + c*akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = A(p,k), aqk = A(q,k);
                    A(p,k) = c*apk - sn*aqk;
                    A(q,k) = sn*apk + c*aqk;
                }
            }
    }
    double lo = A(0,0), hi = A(0,0);
    for (int i = 1; i < n; ++i) { lo = std::min(lo, A(i,i)); hi = std::max(hi, A(i,i)); }
    return {lo, hi};
}

int main() {
    // ---- (1) implicit solve: bond stays inside finite extensibility for any gama ----
    {
        // FENE-Fraenkel reduced natural length q0 = 0.5, so the reduced bond must
        // stay in [0, 1.5] no matter how large the driving gama is.
        const dp dtby4 = 0.01 / 4.0, q0 = 0.5;
        bool bounded = true;
        for (dp g : {1.0e0, 1.0e2, 1.0e4, 1.0e6, 1.0e8, 1.0e10, 1.0e14, 1.0e20}) {
            const dp r = solve_implicit_r(Spring::FENEFraenkel, dtby4, g, q0).length;
            if (!(r >= 0.0 && r <= 1.5) || !std::isfinite(r)) bounded = false;
        }
        check(bounded, "FENE-Fraenkel bond bounded for gama up to 1e20");

        // FENE (reduced natural length 0): bond must stay < 1 for any gama.
        bool fene_ok = true;
        for (dp g : {1.0e0, 1.0e4, 1.0e10, 1.0e20}) {
            const dp r = solve_implicit_r(Spring::FENE, dtby4, g, 0.0).length;
            if (!(r > 0.0 && r < 1.0) || !std::isfinite(r)) fene_ok = false;
        }
        check(fene_ok, "FENE bond bounded (< 1) for gama up to 1e20");
    }

    // ---- (2) Chebyshev square root on a strongly stretched config ----
    {
        const dp hstar = 0.2;
        const Vec3Field R = stretched_chain(30, 2.99);   // near full extension
        const Matrix D = rpy_diffusion_matrix(R, hstar);

        // Worst-case noise: aligned with the smallest-eigenvalue direction. We get a
        // proxy for it cheaply by inverse iteration is overkill here; instead we test
        // several structured X0 (including the all-equal "collective translation"
        // mode, which was the one that blew up) and require the Chebyshev result to
        // satisfy fluctuation-dissipation and stay finite.
        const int n = D.n;
        bool all_fd_ok = true, all_finite = true;
        double worst_fd = 0.0;
        std::vector<Vec3Field> probes;
        { Vec3Field x(30); for (int i=0;i<30;i++) x[i] = {1.0, 1.0, 1.0}; probes.push_back(x); }      // translation mode
        { Vec3Field x(30); for (int i=0;i<30;i++){ dp s=(i%2? -1.0:1.0); x[i]={s,s,s}; } probes.push_back(x); }
        { Vec3Field x(30); for (int i=0;i<30;i++){ x[i]={std::sin(0.9*i),std::cos(0.5*i),std::sin(0.3*i)}; } probes.push_back(x); }

        for (const auto& X0 : probes) {
            const ChebyshevResult cr = chebyshev_sqrt_times(D, X0, 1.0, 2.5e-3);
            worst_fd = std::max(worst_fd, cr.fd_error);
            for (const auto& v : cr.value)
                for (int d = 0; d < 3; ++d)
                    if (!std::isfinite(v[d])) all_finite = false;
            if (cr.fd_error > 2.5e-3) all_fd_ok = false;
        }
        std::printf("  (stretched-config Chebyshev worst fd_error = %.3e)\n", worst_fd);
        check(all_finite, "Chebyshev B.X0 finite on stretched config");
        check(all_fd_ok,  "Chebyshev fluctuation-dissipation on stretched config");

        // squares back to D (symmetric sqrt) for the collective-translation mode
        const Vec3Field X0 = probes[0];
        const ChebyshevResult cr = chebyshev_sqrt_times(D, X0, 1.0, 1e-4);
        const Vec3Field BBx0 = chebyshev_sqrt_times(D, cr.value, 1.0, 1e-4).value;
        const Vec3Field Dx0  = unflatten(matvec(D, flatten(X0)));
        double md = 0.0, scale = 0.0;
        for (size_t m=0;m<R.size();++m) for (int d=0;d<3;++d){
            md = std::max(md, std::fabs(BBx0[m][d]-Dx0[m][d]));
            scale = std::max(scale, std::fabs(Dx0[m][d]));
        }
        std::printf("  (B(B.X0) vs D.X0 rel diff = %.3e)\n", md/scale);
        check(md/scale < 1e-3, "Chebyshev squares back to D on stretched config");
    }

    // ---- (3) full FENE-Fraenkel + HI shear run stays bounded; methods agree ----
    {
        auto run_mean_rg2 = [](DelSMethod method, int seed, bool& blew) {
            PhysParams phys;
            phys.spring.type = Spring::FENEFraenkel;
            phys.spring.sqrtb = 2.0;
            phys.spring.natural_length = 1.0;
            phys.number_of_beads = 30;
            phys.hstar = 0.15;
            phys.hi_method = method;
            phys.flow = Flow::constant(Mat3{{{0.0,1.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}});  // shear rate 1

            // stretched-ish start along x so we probe the dangerous regime quickly
            Vec3Field R = stretched_chain(30, 1.0);
            Rng rng(seed);
            SimParams sim; sim.dt = 0.01; sim.time_start = 0.0;

            blew = false;
            double sum = 0.0; int cnt = 0;
            for (int step = 0; step < 2000; ++step) {
                sim.time_start = step * sim.dt;
                sim.time_end   = sim.time_start;          // exactly one step
                time_integrate_chain(R, phys, sim, rng);
                // max bond
                double mb = 0.0;
                for (size_t i = 1; i < R.size(); ++i) {
                    const Vec3 b = R[i] - R[i-1];
                    mb = std::max(mb, std::sqrt(norm2(b)));
                }
                if (!std::isfinite(mb) || mb > 10.0) { blew = true; break; }
                if (step >= 1000) {  // steady-state sampling
                    Vec3 com{0,0,0}; for (auto& r : R) com = com + r;
                    com = (1.0/R.size()) * com;
                    double rg2 = 0.0; for (auto& r : R){ Vec3 d=r-com; rg2 += norm2(d);} rg2/=R.size();
                    sum += rg2; ++cnt;
                }
            }
            return cnt ? sum / cnt : 0.0;
        };

        int blowups = 0; double sc = 0.0, sk = 0.0; int nc = 0, nk = 0;
        for (int seed = 0; seed < 12; ++seed) {
            bool bc=false, bk=false;
            const double rc = run_mean_rg2(DelSMethod::Chebyshev, seed, bc);
            const double rk = run_mean_rg2(DelSMethod::Cholesky,  seed, bk);
            if (bc || bk) ++blowups;
            if (!bc){ sc += rc; ++nc; }
            if (!bk){ sk += rk; ++nk; }
        }
        check(blowups == 0, "12 FENE-Fraenkel+HI shear runs stay bounded (x2000 steps)");
        const double mc = nc ? sc/nc : 0.0, mk = nk ? sk/nk : 0.0;
        std::printf("  (steady <Rg^2>: Chebyshev=%.2f  Cholesky=%.2f)\n", mc, mk);
        // loose agreement: same physics within 20%
        check(mc > 0 && mk > 0 && std::fabs(mc-mk) < 0.20*mk,
              "Chebyshev and Cholesky agree on steady <Rg^2>");
    }


    // ---- (4) warm-started spectral bounds must still bracket the spectrum ----
    // The Chebyshev interval is built from a Lanczos estimate that is warm-started
    // from the previous step. Cheap, but worthless if it ever fails to contain the
    // spectrum -- the series then amplifies instead of square-rooting. Step a real
    // trajectory and compare the bounds against exact (Jacobi) eigenvalues.
    {
        PhysParams phys;
        phys.spring.type = Spring::FENEFraenkel;
        phys.spring.sqrtb = 2.0;
        phys.spring.natural_length = 1.0;
        phys.number_of_beads = 24;
        phys.hstar = 0.15;
        phys.hi_method = DelSMethod::Chebyshev;
        phys.flow = Flow::constant(Mat3{{{0.0,1.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}});

        Vec3Field R = stretched_chain(24, 1.0);
        Rng rng(11);
        SimParams sim; sim.dt = 0.01;

        SpectralCache cache;          // persists across steps: this is the warm start
        int checked = 0, missed_lo = 0, missed_hi = 0;
        double worst_lo_ratio = 0.0, worst_hi_ratio = 1e9;

        for (int step = 0; step < 300; ++step) {
            sim.time_start = step * sim.dt;
            sim.time_end   = sim.time_start;
            time_integrate_chain(R, phys, sim, rng);

            if (step % 10 == 0) {
                const Matrix D = rpy_diffusion_matrix(R, phys.hstar);
                const auto [lmin, lmax] = spectral_bounds(D, &cache);
                const auto [emin, emax] = jacobi_extremes(D);
                ++checked;
                if (lmin > emin) ++missed_lo;
                if (lmax < emax) ++missed_hi;
                worst_lo_ratio = std::max(worst_lo_ratio, lmin / emin);
                worst_hi_ratio = std::min(worst_hi_ratio, lmax / emax);
            }
        }
        std::printf("  (warm bounds: %d configs, lmin/true_min max = %.3f, "
                    "lmax/true_max min = %.3f; warm runs = %ld, cold = %ld)\n",
                    checked, worst_lo_ratio, worst_hi_ratio,
                    cache.warm_runs, cache.cold_runs);
        check(missed_lo == 0, "warm-started lmin brackets the spectrum below");
        check(missed_hi == 0, "warm-started lmax brackets the spectrum above");
    }

    std::printf("\nHI stability: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
