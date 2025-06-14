#include "mandelbrot/mandelbrot.hpp"
#include <benchmark/benchmark.h>
#include <complex>
#include <format>

// Test cases with their expected iteration behavior
struct TestPoint {
  std::complex<double> point;
  std::string_view name;
};

const TestPoint test_points[] = {
    {{0.0, 0.0}, "WorstCase"},  // Will run full iterations
    {{-0.75, 0.1}, "EdgeCase"}, // Medium iterations
    {{2.0, 2.0}, "BestCase"},   // Will escape quickly
};

template <typename Func>
static void BM_Mandelbrot(benchmark::State &state, Func mandel_func,
                          const TestPoint &test_point) {
  for (auto _ : state) {
    auto result = mandel_func(test_point.point);
    benchmark::DoNotOptimize(result);
  }

  state.counters["calc"] =
      benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_Mandelbrot_SIMD(benchmark::State &state, auto mandel_func,
                               const TestPoint &test_point) {
  using batch = xsimd::batch<double>;
  constexpr std::size_t N = batch::size;

  auto const a = batch(test_point.point.real());
  auto const b = batch(test_point.point.imag());

  for (auto _ : state) {
    auto result = mandel_func(a, b);
    benchmark::DoNotOptimize(result);
  }

  state.counters["calc"] =
      benchmark::Counter(N, benchmark::Counter::kIsIterationInvariantRate);
}

void register_scalar_benchmarks(const char *version_name, auto func) {
  for (const auto &point : test_points) {
    benchmark::RegisterBenchmark(
        std::format("BM_Mandelbrot_{}/{}", version_name, point.name),
        [=](benchmark::State &state) { BM_Mandelbrot(state, func, point); });
  }
}

void register_simd_benchmarks(const char *version_name, auto func) {
  for (const auto &point : test_points) {
    benchmark::RegisterBenchmark(
        std::format("BM_Mandelbrot_{}/{}", version_name, point.name),
        [=](benchmark::State &state) {
          BM_Mandelbrot_SIMD(state, func, point);
        });
  }
}

// Register all benchmark combinations
int main(int argc, char **argv) {

  constexpr auto MAX_ITER = 10'000uz;

  register_scalar_benchmarks("V1", [](std::complex<double> const &c) {
    return mandelbrot::v1::mandelbrot<MAX_ITER>(c);
  });
  register_scalar_benchmarks("V2", [](std::complex<double> const &c) {
    return mandelbrot::v2::mandelbrot<MAX_ITER>(c);
  });
  register_scalar_benchmarks("V3", [](std::complex<double> const &c) {
    return mandelbrot::v3::mandelbrot<MAX_ITER>(c);
  });
  register_scalar_benchmarks("V4", [](std::complex<double> const &c) {
    return mandelbrot::v4::mandelbrot<MAX_ITER>(c);
  });
  register_scalar_benchmarks("V5", [](std::complex<double> const &c) {
    return mandelbrot::v5::mandelbrot<MAX_ITER>(c);
  });
  register_simd_benchmarks("V6",
                           [](xsimd::batch<double> a, xsimd::batch<double> b) {
                             return mandelbrot::v6::mandelbrot<MAX_ITER>(a, b);
                           });

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}