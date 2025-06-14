
#include "mandelbrot/mandelbrot.hpp"
#include <benchmark/benchmark.h>
#include <complex>

// Test cases with their expected iteration behavior
struct TestPoint {
  std::complex<double> point;
  const char *name;
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
}

// Template to create benchmarks for different implementations
template <typename T, std::size_t MAX_ITER>
void register_mandelbrot_benchmarks(const char *version_name, auto func) {
  for (const auto &point : test_points) {
    std::string name =
        std::string("BM_Mandelbrot_") + version_name + "/" + point.name;
    benchmark::RegisterBenchmark(name.c_str(), [=](benchmark::State &state) {
      BM_Mandelbrot(state, func, point);
    });
  }
}

// Register all benchmark combinations
int main(int argc, char **argv) {
  register_mandelbrot_benchmarks<double, 10'000>(
      "V1", [](const std::complex<double> &c) {
        return mandelbrot::v1::mandelbrot<double, 10'000>(c);
      });
  register_mandelbrot_benchmarks<double, 10'000>(
      "V2", [](const std::complex<double> &c) {
        return mandelbrot::v2::mandelbrot<double, 10'000>(c);
      });
  register_mandelbrot_benchmarks<double, 10'000>(
      "V3", [](const std::complex<double> &c) {
        return mandelbrot::v3::mandelbrot<double, 10'000>(c);
      });
  register_mandelbrot_benchmarks<double, 10'000>(
      "V4", [](const std::complex<double> &c) {
        return mandelbrot::v4::mandelbrot<double, 10'000>(c);
      });
  register_mandelbrot_benchmarks<double, 10'000>(
      "V5", [](const std::complex<double> &c) {
        return mandelbrot::v5::mandelbrot<double, 10'000>(c);
      });

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}