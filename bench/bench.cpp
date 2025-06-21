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
static void BM_Mandelbrot(
    benchmark::State &state, Func mandel_func, const std::complex<double> &point
) {
  for (auto _ : state) {
    auto result = mandel_func(point);
    benchmark::DoNotOptimize(result);
  }

  state.counters["calc"] =
      benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_Mandelbrot_SIMD(
    benchmark::State &state, auto mandel_func, const std::complex<double> &point
) {
  using batch = xsimd::batch<double>;
  constexpr std::size_t N = batch::size;

  auto a = batch(point.real());
  auto b = batch(point.imag());

  for (auto _ : state) {
    auto result = mandel_func(a, b);
    benchmark::DoNotOptimize(result);
  }

  state.counters["calc"] =
      benchmark::Counter(N, benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_Mandelbrot_MT(
    benchmark::State &state,
    auto &&mandel_func,
    const std::complex<double> &point
) {

  constexpr std::size_t N = 1024 * 8;

  auto gen = [=](auto i) { return point; };

  // 64KiB
  auto vec = std::vector<size_t>(N);

  // One common thread pool for the whole benchmark
  auto const core_count = std::max(std::thread::hardware_concurrency(), 4u);
  auto pool = exec::static_thread_pool{core_count};
  for (auto _ : state) {
    vec = mandel_func(std::move(vec), gen, pool);
    benchmark::DoNotOptimize(vec);
  }

  state.counters["calc"] =
      benchmark::Counter(N, benchmark::Counter::kIsIterationInvariantRate);
}

void register_scalar_benchmarks(const char *version_name, auto func) {
  for (const auto &point : test_points) {
    benchmark::RegisterBenchmark(
        std::format("BM_Mandelbrot_{}/{}", version_name, point.name),
        [=](benchmark::State &state) {
          BM_Mandelbrot(state, func, point.point);
        }
    );
  }
}

void register_simd_benchmarks(const char *version_name, auto func) {
  for (const auto &point : test_points) {
    benchmark::RegisterBenchmark(
        std::format("BM_Mandelbrot_{}/{}", version_name, point.name),
        [=](benchmark::State &state) {
          BM_Mandelbrot_SIMD(state, func, point.point);
        }
    );
  }
}

void register_mt_benchmarks(const char *version_name, auto func) {
  for (const auto &point : test_points) {
    benchmark::RegisterBenchmark(
        std::format("BM_Mandelbrot_{}/{}", version_name, point.name),
        [=](benchmark::State &state) {
          BM_Mandelbrot_MT(state, func, point.point);
        }
    );
  }
}

// Register all benchmark combinations
int main(int argc, char **argv) {

  constexpr auto MAX_ITER = 10'000uz;
  auto const core_count = std::max(std::thread::hardware_concurrency(), 4u);

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

  // SIMD
  register_simd_benchmarks(
      "V6",
      [](xsimd::batch<double> a, xsimd::batch<double> b) {
        return mandelbrot::v6::mandelbrot<MAX_ITER>(a, b);
      }
  );

  // MT
  register_mt_benchmarks(
      "V7a",
      [core_count](
          std::vector<size_t> vec,
          auto &&gen,
          exec::static_thread_pool & /*ignore*/
      ) {
        // New thread pool per benchmark iteration
        auto pool = exec::static_thread_pool{core_count};
        return mandelbrot::v7::mandelbrot<MAX_ITER>(std::move(vec), gen, pool);
      }
  );

  register_mt_benchmarks(
      "V7b",
      [](std::vector<size_t> vec, auto &&gen, exec::static_thread_pool &pool) {
        // Reuse the same thread pool
        return mandelbrot::v7::mandelbrot<MAX_ITER>(std::move(vec), gen, pool);
      }
  );

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}