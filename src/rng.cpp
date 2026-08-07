#include "rng.hpp"

#include <cmath>    // std::nextafterf
#include <cstdint>

namespace bdsim {

// Constants from ran_1 (utils.f90).
namespace {
constexpr std::int32_t IA = 16807;
constexpr std::int32_t IM = 2147483647;   // 2^31 - 1 = 0x7FFFFFFF
constexpr std::int32_t IQ = 127773;
constexpr std::int32_t IR = 2836;
}

void Rng::maybe_init() {
    // Fortran: If ((seed <= 0) .or. (iy < 0)) Then ... End If
    if (seed_ <= 0 || iy_ < 0) {
        // am = nearest(1.0,-1.0)/real(IM), all single precision.
        // nearest(1.0,-1.0) is the largest float strictly below 1.0f;
        // real(IM) rounds to 2147483648.0f.
        am_ = std::nextafterf(1.0f, -1.0f) / static_cast<float>(IM);

        const std::int32_t s = std::abs(seed_);
        iy_ = (888889999 ^ s) | 1;
        ix_ =  777755555 ^ s;
        seed_ = s + 1;
    }
}

void Rng::ran_1(int n, dp* x) {
    maybe_init();

    // ix is manipulated as an unsigned 32-bit pattern (ishft is a *logical*
    // shift in Fortran, so the right shift must be zero-filling).
    std::uint32_t ux = static_cast<std::uint32_t>(ix_);

    for (int c = 0; c < n; ++c) {
        // Marsaglia xorshift on ix.
        ux ^= (ux << 13);
        ux ^= (ux >> 17);
        ux ^= (ux << 5);

        // Park-Miller minstd via Schrage's method on iy (stays within int32).
        const std::int32_t k = iy_ / IQ;
        iy_ = IA * (iy_ - k * IQ) - IR * k;
        if (iy_ < 0) iy_ += IM;

        // result = ior(iand(IM, ieor(ix,iy)), 1)  in [1, 2^31-1]
        const std::uint32_t mixed = ux ^ static_cast<std::uint32_t>(iy_);
        const std::int32_t result =
            static_cast<std::int32_t>((mixed & static_cast<std::uint32_t>(IM)) | 1u);

        // X = am * result, computed in single precision (am is float), then
        // widened to double — matching the Fortran mixed-kind expression.
        x[c] = static_cast<dp>(am_ * static_cast<float>(result));
    }

    ix_ = static_cast<std::int32_t>(ux);
}

} // namespace bdsim
