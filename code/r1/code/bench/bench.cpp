// CPU-time benchmark: cost of one fixed step of every method on every test
// problem, real arithmetic for the real tableaux and std::complex<double> for
// the projected complex tableaux (the state stays real, stages are complex).
// Output: CSV  problem,method,stages,evals_per_step,ns_per_step,ns_per_f_real,ns_per_f_complex
// R1: up to SMAX = 10 stages (Vern6(5)); the large problem Bruss2D (882 equations) is timed with
// fewer steps and evaluations (NSTEP, NEVAL), the small problems as in the original submission.
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include "coeffs.h"

using cd = std::complex<double>;

// ---------------------------------------------------------------- problems
// NSTEP: steps per timing of a method; NEVAL: evaluations per timing of f (defaults below)
struct Small { static const long NSTEP = 400000; static const long NEVAL = 4000000; };
template <class T> struct Exponential : Small { static const int D = 1; static constexpr double TEND = 5;
  static void f(T t, const T* y, T* d) { (void)t; d[0] = -std::exp(y[0]); }
  static void y0(double* y) { y[0] = 1; } };
template <class T> struct NonAut : Small { static const int D = 1; static constexpr double TEND = 5;
  static void f(T t, const T* y, T* d) { T s = std::sin(t); d[0] = 4.0 * y[0] * s * s * s * std::cos(t); }
  static void y0(double* y) { y[0] = 1; } };
template <class T> struct VdP : Small { static const int D = 2; static constexpr double TEND = 5;
  static void f(T, const T* y, T* d) { d[0] = y[1]; d[1] = (1.0 - y[0] * y[0]) * y[1] - y[0]; }
  static void y0(double* y) { y[0] = 2; y[1] = 0; } };
template <class T> struct Rossler : Small { static const int D = 3; static constexpr double TEND = 5;
  static void f(T, const T* y, T* d) { d[0] = -y[1] - y[2]; d[1] = y[0] + 0.2 * y[1]; d[2] = 0.2 + y[2] * (y[0] - 5.7); }
  static void y0(double* y) { y[0] = 1; y[1] = 1; y[2] = 1; } };
template <class T> struct Lorenz : Small { static const int D = 3; static constexpr double TEND = 5;
  static void f(T, const T* y, T* d) { d[0] = 10.0 * (y[1] - y[0]); d[1] = y[0] * (28.0 - y[2]) - y[1]; d[2] = y[0] * y[1] - (8.0 / 3.0) * y[2]; }
  static void y0(double* y) { y[0] = 1; y[1] = 1; y[2] = 20; } };
template <class T> struct Bouali : Small { static const int D = 3; static constexpr double TEND = 5;
  static void f(T, const T* y, T* d) { d[0] = 3.0 * y[0] * (1.0 - y[1]) - 2.2 * y[2]; d[1] = -y[1] * (1.0 - y[0] * y[0]); d[2] = 0.001 * y[0]; }
  static void y0(double* y) { y[0] = 1; y[1] = 0.1; y[2] = 0; } };
template <class T> struct Kepler : Small { static const int D = 4; static constexpr double TEND = 20;
  static void f(T, const T* y, T* d) { T r2 = y[0] * y[0] + y[1] * y[1]; T r3 = r2 * std::sqrt(r2);
    d[0] = y[2]; d[1] = y[3]; d[2] = -y[0] / r3; d[3] = -y[1] / r3; }
  static void y0(double* y) { y[0] = 0.5; y[1] = 0; y[2] = 0; y[3] = std::sqrt(3.0); } };
template <class T> struct Rigid : Small { static const int D = 3; static constexpr double TEND = 12;
  static void f(T, const T* y, T* d) { d[0] = y[1] * y[2]; d[1] = -y[0] * y[2]; d[2] = -0.51 * y[0] * y[1]; }
  static void y0(double* y) { y[0] = 0; y[1] = 1; y[2] = 1; } };
template <class T> struct Brusselator : Small { static const int D = 2; static constexpr double TEND = 20;
  static void f(T, const T* y, T* d) { T q = y[0] * y[0] * y[1]; d[0] = 1.0 + q - 4.0 * y[0]; d[1] = 3.0 * y[0] - q; }
  static void y0(double* y) { y[0] = 1.5; y[1] = 3; } };
template <class T> struct Arenstorf : Small { static const int D = 4; static constexpr double TEND = 17.0652165601579625588917206249;
  static void f(T, const T* y, T* d) { const double mu = 0.012277471, mp = 1 - mu;
    T a = y[0] + mu, b = y[0] - mp; T r1 = a * a + y[1] * y[1], r2 = b * b + y[1] * y[1];
    T D1 = r1 * std::sqrt(r1), D2 = r2 * std::sqrt(r2);
    d[0] = y[2]; d[1] = y[3]; d[2] = y[0] + 2.0 * y[3] - mp * a / D1 - mu * b / D2;
    d[3] = y[1] - 2.0 * y[2] - mp * y[1] / D1 - mu * y[1] / D2; }
  static void y0(double* y) { y[0] = 0.994; y[1] = 0; y[2] = 0; y[3] = -2.00158510637908252240537862224; } };

// two-dimensional Brusselator with diffusion, N = 21 grid points per direction, Neumann conditions by
// reflection (u_0j = u_2j, u_{N+1}j = u_{N-1}j), five-point Laplacian; y = (u, v), u[i*NG + j] at (x_i, y_j)
template <class T> struct Bruss2D { static const int NG = 21; static const int D = 2 * NG * NG;
  static const long NSTEP = 4000; static const long NEVAL = 40000; static constexpr double TEND = 11.5;
  static void f(T, const T* y, T* d) {
    const double al = 2e-3 * (NG - 1) * (NG - 1);
    const T* u = y; const T* v = y + NG * NG; T* du = d; T* dv = d + NG * NG;
    for (int i = 0; i < NG; ++i) {
      const int im = i == 0 ? 1 : i - 1, ip = i == NG - 1 ? NG - 2 : i + 1;
      for (int j = 0; j < NG; ++j) {
        const int jm = j == 0 ? 1 : j - 1, jp = j == NG - 1 ? NG - 2 : j + 1;
        const int k = i * NG + j;
        const T uk = u[k], vk = v[k], q = uk * uk * vk;
        const T lu = u[im * NG + j] + u[ip * NG + j] + u[i * NG + jm] + u[i * NG + jp] - 4.0 * uk;
        const T lv = v[im * NG + j] + v[ip * NG + j] + v[i * NG + jm] + v[i * NG + jp] - 4.0 * vk;
        du[k] = 1.0 + q - 4.4 * uk + al * lu;
        dv[k] = 3.4 * uk - q + al * lv;
      } } }
  static void y0(double* y) { for (int i = 0; i < NG; ++i) for (int j = 0; j < NG; ++j) {
    y[i * NG + j] = 0.5 + j / double(NG - 1); y[NG * NG + i * NG + j] = 1.0 + 5.0 * i / double(NG - 1); } } };

// ---------------------------------------------------------------- integrators
volatile double sink;

template <template <class> class P>
double run_real(const Tab& tb, long N) {
  const int D = P<double>::D; const int s = tb.s;
  double y[D], K[SMAX][D], Y[D]; P<double>::y0(y);
  const double h = P<double>::TEND / N; double t = 0;
  bool have_k1 = false;
  for (long n = 0; n < N; ++n) {
    if (!(tb.fsal && have_k1)) P<double>::f(t, y, K[0]);
    for (int i = 1; i < s; ++i) {
      for (int q = 0; q < D; ++q) { double acc = 0; for (int j = 0; j < i; ++j) if (tb.are[i][j] != 0) acc += tb.are[i][j] * K[j][q]; Y[q] = y[q] + h * acc; }
      P<double>::f(t + tb.cre[i] * h, Y, K[i]);
    }
    if (tb.fsal) { for (int q = 0; q < D; ++q) y[q] = Y[q]; for (int q = 0; q < D; ++q) K[0][q] = K[s - 1][q]; have_k1 = true; }
    else for (int q = 0; q < D; ++q) { double acc = 0; for (int i = 0; i < s; ++i) if (tb.bre[i] != 0) acc += tb.bre[i] * K[i][q]; y[q] += h * acc; }
    t += h;
  }
  double r = 0; for (int q = 0; q < D; ++q) r += y[q]; return r;
}

template <template <class> class P>
double run_cplx(const Tab& tb, long N) {
  const int D = P<cd>::D; const int s = tb.s;
  double y[D]; cd K[SMAX][D], Y[D]; P<cd>::y0(y);
  cd a[SMAX][SMAX], b[SMAX], c[SMAX];
  for (int i = 0; i < SMAX; ++i) { b[i] = cd(tb.bre[i], tb.bim[i]); c[i] = cd(tb.cre[i], tb.cim[i]);
    for (int j = 0; j < SMAX; ++j) a[i][j] = cd(tb.are[i][j], tb.aim[i][j]); }
  const double h = P<cd>::TEND / N; double t = 0;
  double yr[D], k1[D], tmp[D];
  for (long n = 0; n < N; ++n) {
    // first stage at the real point: a real evaluation
    for (int q = 0; q < D; ++q) yr[q] = y[q];
    { P<double>::f(t, yr, tmp); for (int q = 0; q < D; ++q) { k1[q] = tmp[q]; K[0][q] = cd(tmp[q], 0); } }
    for (int i = 1; i < s; ++i) {
      for (int q = 0; q < D; ++q) { cd acc = a[i][0] * k1[q]; for (int j = 1; j < i; ++j) acc += a[i][j] * K[j][q]; Y[q] = y[q] + h * acc; }
      P<cd>::f(cd(t, 0) + c[i] * h, Y, K[i]);
    }
    for (int q = 0; q < D; ++q) { double acc = b[0].real() * k1[q]; for (int i = 1; i < s; ++i) acc += (b[i] * K[i][q]).real(); y[q] += h * acc; }
    t += h;
  }
  double r = 0; for (int q = 0; q < D; ++q) r += y[q]; return r;
}

template <template <class> class P>
double time_f(bool cplx, long M) {
  const int D = P<double>::D;
  auto t0 = std::chrono::high_resolution_clock::now();
  if (cplx) { cd y[D], d[D]; double y0[D]; P<double>::y0(y0);
    for (int q = 0; q < D; ++q) y[q] = cd(y0[q], 1e-3 * (q + 1));
    for (long m = 0; m < M; ++m) { P<cd>::f(cd(0.1, 0.01), y, d); for (int q = 0; q < D; ++q) y[q] += 1e-12 * d[q]; }
    sink = y[0].real();
  } else { double y[D], d[D]; P<double>::y0(y);
    for (long m = 0; m < M; ++m) { P<double>::f(0.1, y, d); for (int q = 0; q < D; ++q) y[q] += 1e-12 * d[q]; }
    sink = y[0]; }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / M;
}

template <template <class> class P>
void bench(const char* pname) {
  const long M = P<double>::NEVAL;
  double fr = 1e30, fc = 1e30;
  for (int rep = 0; rep < 5; ++rep) { fr = std::fmin(fr, time_f<P>(false, M)); fc = std::fmin(fc, time_f<P>(true, M)); }
  for (int k = 0; k < NMETH; ++k) {
    const Tab& tb = TABS[k];
    const long N = P<double>::NSTEP;
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
      auto t0 = std::chrono::high_resolution_clock::now();
      sink = tb.cplx ? run_cplx<P>(tb, N) : run_real<P>(tb, N);
      auto t1 = std::chrono::high_resolution_clock::now();
      best = std::fmin(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / N);
    }
    int ev = tb.fsal ? tb.s - 1 : tb.s;
    std::printf("%s,%s,%d,%d,%.3f,%.3f,%.3f\n", pname, tb.name, tb.s, ev, best, fr, fc);
  }
}

#ifdef CHECK
// correctness check (build with /DCHECK): sum of the components of the solution of Bruss2D after
// 2000 steps for every method, to be compared with the Python implementation (bruss2d_r1.py)
int main() {
  for (int k = 0; k < NMETH; ++k) { const Tab& tb = TABS[k];
    std::printf("%s,%.15e\n", tb.name, tb.cplx ? run_cplx<Bruss2D>(tb, 2000) : run_real<Bruss2D>(tb, 2000)); }
  return 0;
}
#else
int main() {
  std::printf("problem,method,stages,evals_per_step,ns_per_step,ns_per_f_real,ns_per_f_complex\n");
  bench<Exponential>("exponential"); bench<NonAut>("nonautonomous"); bench<VdP>("VanderPol");
  bench<Rossler>("Rossler"); bench<Lorenz>("Lorenz"); bench<Bouali>("Bouali"); bench<Kepler>("Kepler");
  bench<Rigid>("RigidBody"); bench<Brusselator>("Brusselator"); bench<Arenstorf>("Arenstorf");
  bench<Bruss2D>("Bruss2D");
  return 0;
}
#endif
