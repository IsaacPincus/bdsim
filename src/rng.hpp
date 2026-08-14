// rng.hpp — bit-exact port of the Numerical Recipes RNG in utils.f90
// (module random_numbers: ran_1, GaussRand, seed handling).
//
// Fidelity notes (why this is not just std::mt19937):
//   * The Fortran generator combines a Marsaglia xorshift (ix) with a
//     Park-Miller minstd via Schrage's method (iy). We reproduce the exact
//     32-bit integer arithmetic so regression tests match to 1e-6.
//   * `am`, the scale factor mapping the 31-bit integer into (0,1), is a
//     *single precision* real in Fortran. That makes the stream single-precision
//     granular. We keep `float` here on purpose — do not "upgrade" it.
//     It is a pure constant and deliberately does NOT live in the object: it
//     used to be a member set only by the lazy initialiser, so a restored state
//     (seed > 0, iy > 0) skipped initialisation, left it zero, and the generator
//     silently returned an all-zero stream. See rng.cpp.
//   * State (ix, iy, seed) is encapsulated in an object rather than global
//     module state. reset()/save()/restore() cover the seed-reset and
//     variance-reduction stream-replay use cases from the Fortran, and are
//     what the Python bindings pickle through.
#pragma once

#include "types.hpp"

namespace bdsim {

class Rng {
public:
    // Mirrors reset_RNG_with_seed(seed_in [, ix_in, iy_in]).
    // Defaults ix=1, iy=-1 force lazy re-initialisation on the next draw
    // (that is how the Fortran seeds a fresh stream).
    explicit Rng(i32 seed_in = 123) { reset(seed_in); }

    void reset(i32 seed_in, i32 ix_in = 1, i32 iy_in = -1) {
        ix_ = ix_in;
        iy_ = iy_in;
        seed_ = seed_in;
    }

    // Fortran set_seed: change only the seed, leave ix/iy as-is.
    void set_seed(i32 seed_in) { seed_ = seed_in; }

    // Snapshot / restore of the full internal state (get_all_parameters +
    // reset_RNG_with_seed with explicit ix,iy) — used for variance reduction,
    // where two chains must consume an identical random stream.
    // State is the *whole* mutable state -- a restored generator continues the
    // saved stream exactly (test_rng checks this round trip).
    struct State { i32 ix, iy, seed; };
    State save() const { return {ix_, iy_, seed_}; }
    void restore(const State& s) { ix_ = s.ix; iy_ = s.iy; seed_ = s.seed; }

    // ran_1: fill x[0..n) with uniform deviates in (0,1).
    void ran_1(int n, dp* x);

    // Single uniform deviate in (0,1).
    dp next() { dp v; ran_1(1, &v); return v; }

private:
    i32 ix_;
    i32 iy_;
    i32 seed_;
    // No `am_` member: it is a constant, not state. See the header note.

    void maybe_init();
};

} // namespace bdsim
