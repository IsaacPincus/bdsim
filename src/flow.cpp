#include "flow.hpp"

namespace bdsim {

Mat3 Flow::kappa(dp t) const {
    if (constant_ || times_.empty()) return K0_;

    const int n = static_cast<int>(times_.size());
    if (n == 1 || t <= times_.front()) return Ks_.front();
    if (t >= times_.back()) return Ks_.back();

    // find bracket: times_[i] <= t < times_[i+1]
    int i = 0;
    // simple ascending search; tables are typically short
    while (i < n - 1 && !(t >= times_[i] && t < times_[i + 1])) ++i;

    const dp x0 = times_[i], x1 = times_[i + 1];
    const dp w = (t - x0) / (x1 - x0);
    const Mat3& A = Ks_[i];
    const Mat3& B = Ks_[i + 1];
    Mat3 K{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            K[r][c] = A[r][c] + w * (B[r][c] - A[r][c]);
    return K;
}

} // namespace bdsim
