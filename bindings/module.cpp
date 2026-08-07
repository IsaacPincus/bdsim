// module.cpp — nanobind bindings for the bdsim core.
//
// Exposes the parameter structs, enums, flow, RNG, the integrator, and the
// individual force calculations. Parameter objects are picklable so they can be
// shipped to worker processes / MPI ranks for parallel ensembles. Bead positions
// and forces cross the boundary as (N, 3) float64 numpy arrays.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "config.hpp"
#include "integrator.hpp"
#include "model.hpp"

#include <tuple>
#include <vector>

namespace nb = nanobind;
using namespace bdsim;

namespace {

using RowMajor2D = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using NumpyOut   = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

Vec3Field to_field(const RowMajor2D& a) {
    const size_t N = a.shape(0);
    const double* d = a.data();
    Vec3Field R(N);
    for (size_t i = 0; i < N; ++i) R[i] = {d[3 * i], d[3 * i + 1], d[3 * i + 2]};
    return R;
}

NumpyOut to_numpy(const Vec3Field& R) {
    const size_t N = R.size();
    double* data = new double[3 * N];
    for (size_t i = 0; i < N; ++i)
        for (int k = 0; k < 3; ++k) data[3 * i + k] = R[i][k];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const size_t shape[2] = {N, 3};
    return NumpyOut(data, 2, shape, owner);
}

NumpyOut integrate(RowMajor2D R_in, PhysParams phys, const SimParams& sim, Rng& rng) {
    Vec3Field R = to_field(R_in);
    phys.number_of_beads = static_cast<int>(R.size());
    time_integrate_chain(R, phys, sim, rng);
    return to_numpy(R);
}

NumpyOut spring_force_of(RowMajor2D R_in, const SpringParams& sp) {
    Vec3Field R = to_field(R_in);
    return to_numpy(spring_force(chain_geometry(R), sp));
}
NumpyOut bending_force_of(RowMajor2D R_in, const BendingParams& bp) {
    Vec3Field R = to_field(R_in);
    ChainGeometry g = chain_geometry(R);
    return to_numpy(BendingForce(bp).force({R, g}));
}
NumpyOut ev_force_of(RowMajor2D R_in, const EVParams& ep, bool equilibration) {
    Vec3Field R = to_field(R_in);
    ChainGeometry g = chain_geometry(R);
    ExcludedVolume ev(ep, equilibration, static_cast<int>(R.size()));
    return to_numpy(ev.force({R, g}));
}
NumpyOut total_force_of(RowMajor2D R_in, PhysParams phys) {
    Vec3Field R = to_field(R_in);
    phys.number_of_beads = static_cast<int>(R.size());
    PhysicalModel model(phys);
    ChainGeometry g = chain_geometry(R);
    Vec3Field F = spring_force(g, model.spring());
    const Vec3Field Fe = model.non_spring_force({R, g});
    for (size_t i = 0; i < F.size(); ++i) F[i] = F[i] + Fe[i];
    return to_numpy(F);
}

Vec3 to_vec3(std::tuple<dp, dp, dp> t) {
    return Vec3{std::get<0>(t), std::get<1>(t), std::get<2>(t)};
}

// --- ExternalForce (de)serialisation for pickling ---
// Flattened as parallel arrays so the state is a plain tuple of vectors.
std::tuple<std::vector<int>, std::vector<int>, std::vector<dp>,
           std::vector<dp>, std::vector<int>, std::vector<dp>>
external_getstate(const ExternalForce& e) {
    std::vector<int> beads, is_const, ntimes;
    std::vector<dp> consts, times, values;
    for (const BeadForce& b : e.entries()) {
        beads.push_back(b.bead);
        is_const.push_back(b.constant ? 1 : 0);
        for (int i = 0; i < 3; ++i) consts.push_back(b.value[i]);
        ntimes.push_back(static_cast<int>(b.times.size()));
        for (dp t : b.times) times.push_back(t);
        for (const Vec3& v : b.values)
            for (int i = 0; i < 3; ++i) values.push_back(v[i]);
    }
    return {beads, is_const, consts, times, ntimes, values};
}

ExternalForce external_setstate(const std::vector<int>& beads,
                                const std::vector<int>& is_const,
                                const std::vector<dp>& consts,
                                const std::vector<dp>& times,
                                const std::vector<int>& ntimes,
                                const std::vector<dp>& values) {
    ExternalForce e;
    std::vector<BeadForce> entries;
    size_t ti = 0, vi = 0;
    for (size_t k = 0; k < beads.size(); ++k) {
        BeadForce b;
        b.bead = beads[k];
        b.constant = is_const[k] != 0;
        b.value = Vec3{consts[3 * k], consts[3 * k + 1], consts[3 * k + 2]};
        const int n = ntimes[k];
        for (int j = 0; j < n; ++j) {
            b.times.push_back(times[ti + j]);
            b.values.push_back(Vec3{values[vi + 3 * j], values[vi + 3 * j + 1],
                                    values[vi + 3 * j + 2]});
        }
        ti += n; vi += 3 * n;
        entries.push_back(std::move(b));
    }
    e.set_entries(std::move(entries));
    return e;
}

// External force on every bead at time t, for inspection from Python.
NumpyOut external_force_of(PhysParams phys, int nbeads, dp t) {
    phys.number_of_beads = nbeads;
    return to_numpy(PhysicalModel(phys).external_force(t));
}

Flow flow_constant(RowMajor2D K) {
    const double* d = K.data();
    Mat3 m{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m[i][j] = d[3 * i + j];
    return Flow::constant(m);
}

// --- Flow (de)serialisation for pickling ---
std::tuple<bool, std::vector<dp>, std::vector<dp>, std::vector<dp>> flow_getstate(const Flow& f) {
    std::vector<dp> k0(9);
    const Mat3& m = f.constant_kappa();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) k0[3 * i + j] = m[i][j];
    std::vector<dp> ks;
    for (const Mat3& M : f.sample_tensors())
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) ks.push_back(M[i][j]);
    return {f.is_constant(), k0, f.sample_times(), ks};
}
Flow flow_setstate(bool constant, const std::vector<dp>& k0,
                   const std::vector<dp>& times, const std::vector<dp>& ks) {
    if (constant) {
        Mat3 m{};
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m[i][j] = k0[3 * i + j];
        return Flow::constant(m);
    }
    std::vector<Mat3> Ks(times.size());
    for (size_t t = 0; t < times.size(); ++t)
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Ks[t][i][j] = ks[9 * t + 3 * i + j];
    return Flow::time_varying(times, Ks);
}

}  // namespace

NB_MODULE(_bdsim, m) {
    m.doc() = "Brownian dynamics bead-spring simulation core";

    nb::enum_<Spring>(m, "Spring")
        .value("Hook", Spring::Hook).value("FENE", Spring::FENE)
        .value("ILC", Spring::ILC).value("WLC", Spring::WLC)
        .value("Fraenkel", Spring::Fraenkel).value("FENEFraenkel", Spring::FENEFraenkel)
        .value("WLCbounded", Spring::WLCbounded);
    nb::enum_<EV>(m, "EV")
        .value("None_", EV::None).value("Gauss", EV::Gauss)
        .value("LJ", EV::LJ).value("SDK", EV::SDK).value("SDKstickers", EV::SDKstickers);
    nb::enum_<Bending>(m, "Bending")
        .value("None_", Bending::None).value("OneMinusCosTheta", Bending::OneMinusCosTheta);
    nb::enum_<DelSMethod>(m, "DelSMethod")
        .value("Chebyshev", DelSMethod::Chebyshev).value("Cholesky", DelSMethod::Cholesky)
        .value("ExactSqrt", DelSMethod::ExactSqrt);

    nb::class_<SpringParams>(m, "SpringParams")
        .def(nb::init<>())
        .def_rw("type", &SpringParams::type)
        .def_rw("sqrtb", &SpringParams::sqrtb)
        .def_rw("natural_length", &SpringParams::natural_length)
        .def("__getstate__", [](const SpringParams& p) {
            return std::make_tuple(static_cast<int>(p.type), p.sqrtb, p.natural_length); })
        .def("__setstate__", [](SpringParams& p, std::tuple<int, dp, dp> s) {
            new (&p) SpringParams{static_cast<Spring>(std::get<0>(s)),
                                  std::get<1>(s), std::get<2>(s)}; });

    nb::class_<EVParams>(m, "EVParams")
        .def(nb::init<>())
        .def_rw("type", &EVParams::type).def_rw("zstar", &EVParams::zstar)
        .def_rw("dstar", &EVParams::dstar).def_rw("min_cutoff", &EVParams::min_cutoff)
        .def_rw("max_cutoff", &EVParams::max_cutoff)
        .def_rw("contour_dist_for_EV", &EVParams::contour_dist_for_EV)
        .def_rw("phi", &EVParams::phi)
        .def("__getstate__", [](const EVParams& p) {
            return std::make_tuple(static_cast<int>(p.type), p.zstar, p.dstar, p.min_cutoff,
                                   p.max_cutoff, p.contour_dist_for_EV, p.phi); })
        .def("__setstate__", [](EVParams& p,
                std::tuple<int, dp, dp, dp, dp, int, std::vector<dp>> s) {
            new (&p) EVParams();
            p.type = static_cast<EV>(std::get<0>(s)); p.zstar = std::get<1>(s);
            p.dstar = std::get<2>(s); p.min_cutoff = std::get<3>(s);
            p.max_cutoff = std::get<4>(s); p.contour_dist_for_EV = std::get<5>(s);
            p.phi = std::get<6>(s); });

    nb::class_<BendingParams>(m, "BendingParams")
        .def(nb::init<>())
        .def_rw("type", &BendingParams::type).def_rw("stiffness", &BendingParams::stiffness)
        .def("__getstate__", [](const BendingParams& p) {
            return std::make_tuple(static_cast<int>(p.type), p.stiffness); })
        .def("__setstate__", [](BendingParams& p, std::tuple<int, dp> s) {
            new (&p) BendingParams{static_cast<Bending>(std::get<0>(s)), std::get<1>(s)}; });

    nb::class_<Flow>(m, "Flow")
        .def(nb::init<>())
        .def_static("constant", &flow_constant, nb::arg("kappa"))
        .def("__getstate__", &flow_getstate)
        .def("__setstate__", [](Flow& f,
                std::tuple<bool, std::vector<dp>, std::vector<dp>, std::vector<dp>> s) {
            new (&f) Flow(flow_setstate(std::get<0>(s), std::get<1>(s),
                                        std::get<2>(s), std::get<3>(s))); });

    nb::class_<ExternalForce>(m, "ExternalForce",
        "User-applied forces on individual beads (constant or time-varying).\n"
        "Negative bead indices count from the end, so -1 is the last bead.")
        .def(nb::init<>())
        .def("add_constant", [](ExternalForce& e, int bead, std::tuple<dp, dp, dp> F) {
                e.add_constant(bead, to_vec3(F)); },
             nb::arg("bead"), nb::arg("force"),
             "Apply a constant force vector to one bead.")
        .def("add_time_varying", [](ExternalForce& e, int bead, std::vector<dp> times,
                                    std::vector<std::tuple<dp, dp, dp>> vals) {
                std::vector<Vec3> v;
                for (auto& x : vals) v.push_back(to_vec3(x));
                e.add_time_varying(bead, std::move(times), std::move(v)); },
             nb::arg("bead"), nb::arg("times"), nb::arg("forces"),
             "Apply a force interpolated linearly between (time, vector) samples;\n"
             "held constant outside the tabulated range.")
        .def("add_stretch", [](ExternalForce& e, int bead_minus, int bead_plus,
                               std::tuple<dp, dp, dp> F) {
                e.add_stretch(bead_minus, bead_plus, to_vec3(F)); },
             nb::arg("bead_minus"), nb::arg("bead_plus"), nb::arg("force"),
             "Equal and opposite forces: +force on bead_plus, -force on bead_minus.\n"
             "Zero net force, so the chain is stretched without drifting.")
        .def("empty", &ExternalForce::empty)
        .def("__getstate__", &external_getstate)
        .def("__setstate__", [](ExternalForce& e,
                std::tuple<std::vector<int>, std::vector<int>, std::vector<dp>,
                           std::vector<dp>, std::vector<int>, std::vector<dp>> s) {
            new (&e) ExternalForce(external_setstate(std::get<0>(s), std::get<1>(s),
                std::get<2>(s), std::get<3>(s), std::get<4>(s), std::get<5>(s))); });

    nb::class_<PhysParams>(m, "PhysParams")
        .def(nb::init<>())
        .def_rw("spring", &PhysParams::spring).def_rw("ev", &PhysParams::ev)
        .def_rw("bend", &PhysParams::bend).def_rw("flow", &PhysParams::flow)
        .def_rw("external", &PhysParams::external)
        .def_rw("number_of_beads", &PhysParams::number_of_beads)
        .def_rw("hstar", &PhysParams::hstar).def_rw("hi_method", &PhysParams::hi_method)
        .def_rw("ncheb_multiplier", &PhysParams::ncheb_multiplier)
        .def_rw("fd_err_max", &PhysParams::fd_err_max)
        .def_rw("equilibration", &PhysParams::equilibration)
        .def("__getstate__", [](const PhysParams& p) {
            return std::make_tuple(p.spring, p.ev, p.bend, p.flow, p.number_of_beads,
                                   p.hstar, static_cast<int>(p.hi_method),
                                   p.ncheb_multiplier, p.fd_err_max, p.equilibration,
                                   p.external); })
        .def("__setstate__", [](PhysParams& p,
                std::tuple<SpringParams, EVParams, BendingParams, Flow, int, dp, int, dp, dp,
                           bool, ExternalForce> s) {
            new (&p) PhysParams();
            p.spring = std::get<0>(s); p.ev = std::get<1>(s); p.bend = std::get<2>(s);
            p.flow = std::get<3>(s); p.number_of_beads = std::get<4>(s);
            p.hstar = std::get<5>(s); p.hi_method = static_cast<DelSMethod>(std::get<6>(s));
            p.ncheb_multiplier = std::get<7>(s); p.fd_err_max = std::get<8>(s);
            p.equilibration = std::get<9>(s); p.external = std::get<10>(s); });

    nb::class_<SimParams>(m, "SimParams")
        .def(nb::init<>())
        .def_rw("time_start", &SimParams::time_start).def_rw("time_end", &SimParams::time_end)
        .def_rw("dt", &SimParams::dt)
        .def_rw("implicit_loop_tol", &SimParams::implicit_loop_tol)
        .def_rw("update_center_of_mass", &SimParams::update_center_of_mass)
        .def("__getstate__", [](const SimParams& s) {
            return std::make_tuple(s.time_start, s.time_end, s.dt, s.implicit_loop_tol,
                                   s.update_center_of_mass); })
        .def("__setstate__", [](SimParams& s, std::tuple<dp, dp, dp, dp, bool> t) {
            new (&s) SimParams();
            s.time_start = std::get<0>(t); s.time_end = std::get<1>(t); s.dt = std::get<2>(t);
            s.implicit_loop_tol = std::get<3>(t); s.update_center_of_mass = std::get<4>(t); });

    nb::class_<Rng>(m, "Rng")
        .def(nb::init<i32>(), nb::arg("seed") = 123)
        .def("reset", [](Rng& r, i32 seed) { r.reset(seed); }, nb::arg("seed"))
        .def("__getstate__", [](const Rng& r) {
            auto st = r.save(); return std::make_tuple(st.ix, st.iy, st.seed); })
        .def("__setstate__", [](Rng& r, std::tuple<i32, i32, i32> s) {
            new (&r) Rng();
            r.restore({std::get<0>(s), std::get<1>(s), std::get<2>(s)}); });

    m.def("integrate", &integrate, nb::arg("R"), nb::arg("phys"), nb::arg("sim"), nb::arg("rng"),
          "Integrate a chain; returns the evolved (N,3) positions.");
    m.def("spring_force", &spring_force_of, nb::arg("R"), nb::arg("spring"));
    m.def("bending_force", &bending_force_of, nb::arg("R"), nb::arg("bend"));
    m.def("ev_force", &ev_force_of, nb::arg("R"), nb::arg("ev"), nb::arg("equilibration") = false);
    m.def("total_force", &total_force_of, nb::arg("R"), nb::arg("phys"),
          "Intramolecular force (spring + bending + EV). Excludes external forces.");
    m.def("external_force", &external_force_of, nb::arg("phys"), nb::arg("nbeads"),
          nb::arg("t") = 0.0, "Applied external force on each bead at time t.");
}
