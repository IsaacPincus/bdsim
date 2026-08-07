// test_direction.cpp — the connector solve preserves direction.
//
// Appendix statement: "Since both F^(c) and Q are aligned along the same axis,
// the direction of Q must be the same as the direction of Gamma." The solver
// returns Q = (positive scalar) * Gamma, so Q should be parallel to and point
// the same way as Gamma, to near machine precision.
#include "../src/spring.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace bdsim;

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

int main() {
    struct Case { Spring type; dp sqrtb; dp natlen; };
    const std::vector<Case> cases = {
        {Spring::Hook, 10.0, 0.0},   {Spring::FENE, 10.0, 0.0},
        {Spring::ILC, 10.0, 0.0},    {Spring::WLC, 10.0, 0.0},
        {Spring::Fraenkel, 10.0, 3.0}, {Spring::FENEFraenkel, 10.0, 3.0},
        {Spring::WLCbounded, 10.0, 0.5}};
    const std::vector<Vec3> gammas = {
        {1.0, 0.0, 0.0}, {0.0, -2.0, 0.0}, {1.0, 1.0, 1.0},
        {-3.0, 2.0, -1.0}, {0.3, -0.7, 2.4}, {5.0, -4.0, 0.1}};
    const std::vector<dp> dts = {0.01, 0.1, 0.5};

    int checks = 0, failures = 0;
    dp worst_sin = 0.0, min_cos = 2.0;

    for (const auto& c : cases) {
        SpringParams sp{c.type, c.sqrtb, c.natlen};
        for (const auto& g : gammas) {
            for (dp dt : dts) {
                const Vec3 Q = solve_connector(g, dt, sp);
                ++checks;
                // parallel: |Q x g| / (|Q| |g|) ~ sin(angle) ~ 0
                const dp s = norm(cross(Q, g)) / (norm(Q) * norm(g));
                // same direction: Q . g > 0
                const dp cosang = dot(Q, g) / (norm(Q) * norm(g));
                worst_sin = std::max(worst_sin, s);
                min_cos = std::min(min_cos, cosang);
                if (s > 1e-12 || cosang <= 0.0) {
                    ++failures;
                    std::printf("  [FAIL] type=%d dt=%g gamma=(%g,%g,%g): sin=%.2e cos=%.3f\n",
                                (int)c.type, dt, g[0], g[1], g[2], s, cosang);
                }
            }
        }
    }
    std::printf("Direction preserved: %d cases, worst |sin angle| = %.2e, "
                "min cos = %.6f\n", checks, worst_sin, min_cos);
    std::printf("\n%s\n", failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
