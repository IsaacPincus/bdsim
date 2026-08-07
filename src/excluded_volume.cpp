#include "excluded_volume.hpp"

#include <cmath>

namespace bdsim {

namespace {
constexpr dp LJ_REP = 12.0;      // repulsive power
constexpr dp LJ_ATT = 6.0;       // attractive power
constexpr dp SDK_ALPHA = 1.530633312;
constexpr dp SDK_BETA  = 1.213115524;
}

ExcludedVolume::ExcludedVolume(const EVParams& p, bool equilibration, int nbeads)
    : type_(p.type), cd_(p.contour_dist_for_EV), equilibration_(equilibration) {
    const dp z = p.zstar, d = p.dstar;

    phi_.assign(nbeads * nbeads, 0.0);
    if (p.type == EV::SDK && !p.phi.empty()) phi_ = p.phi;

    zsbyds5_ = z / std::pow(d, 5.0);          // Gauss
    pt5bydssq_ = 0.5 / (d * d);

    LJa_ = LJ_REP * 4.0 * z * std::pow(d, LJ_REP);   // LJ
    LJb_ = LJ_ATT * 4.0 * z * std::pow(d, LJ_ATT);

    SDKa_ = LJ_REP * 4.0 * std::pow(d, LJ_REP);      // SDK
    SDKb_ = LJ_ATT * 4.0 * std::pow(d, LJ_ATT);

    Rmin_ = p.min_cutoff;
    Rcutp_ = p.max_cutoff;
    RcutpSDK_ = p.max_cutoff;
    Rcutg_ = d * std::pow(2.0, 1.0 / 6.0);
}

dp ExcludedVolume::gauss_factor(dp r) const {
    return zsbyds5_ * std::exp(-r * r * pt5bydssq_);
}

dp ExcludedVolume::lj_factor(dp r) const {
    const dp Rcut = equilibration_ ? Rcutg_ : Rcutp_;
    if (r <= Rcut && r >= Rmin_)
        return LJa_ / std::pow(r, LJ_REP + 2.0) - LJb_ / std::pow(r, LJ_ATT + 2.0);
    if (r < Rmin_)
        return LJa_ / std::pow(Rmin_, LJ_REP + 2.0) - LJb_ / std::pow(Rmin_, LJ_ATT + 2.0);
    return 0.0;
}

dp ExcludedVolume::sdk_factor(dp r, dp phi_pair) const {
    dp fac = 0.0;
    if (r <= Rcutg_ && r >= Rmin_)
        fac = SDKa_ / std::pow(r, LJ_REP + 2.0) - SDKb_ / std::pow(r, LJ_ATT + 2.0);
    else if (r < Rmin_)
        fac = SDKa_ / std::pow(Rmin_, LJ_REP + 2.0) - SDKb_ / std::pow(Rmin_, LJ_ATT + 2.0);
    if (!equilibration_ && r <= RcutpSDK_ && r > Rcutg_)
        fac += SDK_ALPHA * phi_pair * std::sin(SDK_ALPHA * r * r + SDK_BETA);
    return fac;
}

Vec3Field ExcludedVolume::force(const ChainState& state) const {
    const Vec3Field& R = state.R;
    const int N = static_cast<int>(R.size());
    Vec3Field Fev(N, Vec3{0.0, 0.0, 0.0});
    if (type_ == EV::None) return Fev;

    for (int nu = 1; nu < N; ++nu) {
        for (int mu = 0; mu < nu; ++mu) {
            if ((nu - mu) < cd_) continue;

            const Vec3 q = R[nu] - R[mu];       // separation vector mu -> nu
            dp sq = norm2(q);
            if (sq < 1.0e-12) sq = 1.0e-12;
            const dp r = std::sqrt(sq);

            dp fac = 0.0;
            switch (type_) {
                case EV::Gauss: fac = gauss_factor(r); break;
                case EV::LJ:    fac = lj_factor(r); break;
                case EV::SDK:   fac = sdk_factor(r, phi_[mu * N + nu]); break;
                default:        fac = 0.0; break;  // SDK_stickers: deferred
            }

            const Vec3 f = (-fac) * q;
            Fev[mu] = Fev[mu] + f;
            Fev[nu] = Fev[nu] - f;
        }
    }
    return Fev;
}

} // namespace bdsim
