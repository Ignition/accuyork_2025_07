#include "mandelbrot/mandelbrot.hpp"
#include <benchmark/benchmark.h>
#include <complex>
#include <format>

#include <exec/static_thread_pool.hpp>

struct TestPoint {
  std::complex<double> point;
  std::string_view name;
};

constexpr TestPoint test_points[] = {
    {{0.0, 0.0}, "WorstCase"},  // Will run full iterations
    {{-0.75, 0.1}, "EdgeCase"}, // Medium iterations
    {{2.0, 2.0}, "BestCase"},   // Will escape quickly
};

constexpr auto MAX_ITER = 10'000uz;
constexpr auto PIXEL_COUNT = 1920uz * 1080uz;
static auto THREAD_COUNT = std::thread::hardware_concurrency();
using data_batch = xsimd::batch<size_t>;

/// Global state
static std::unique_ptr<exec::static_thread_pool> pool;
static std::vector<std::size_t> data;
static std::vector<data_batch> data_simd;

/// Setup and Teardown
static void MTSetup(const benchmark::State &state) {
  auto const N = state.range(1);
  auto const thread_count = state.range(2);

  pool = std::make_unique<exec::static_thread_pool>(thread_count);
  data.resize(N);
  auto const N_plus = (N + data_batch::size - 1) / data_batch::size;
  data_simd.resize(N_plus);
}
static void MTTeardown(const benchmark::State &state) { pool.reset(); }

/// Benchmarks
static void BM_Mandelbrot_V1(benchmark::State &state) {
  auto const &test_point = test_points[state.range(0)];
  auto c = test_point.point;
  state.SetLabel(std::format("Naïve [{}]", test_point.name));
  for (auto _ : state) {
    auto result = mandelbrot::v1::mandelbrot<MAX_ITER>(c);
    benchmark::DoNotOptimize(result);
  }
  state.counters["calc"] = benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_V1)->DenseRange(0, 2);

static void BM_Mandelbrot_V2(benchmark::State &state) {
  auto const &test_point = test_points[state.range(0)];
  auto c = test_point.point;
  state.SetLabel(std::format("Without sqrt [{}]", test_point.name));
  for (auto _ : state) {
    auto result = mandelbrot::v2::mandelbrot<MAX_ITER>(c);
    benchmark::DoNotOptimize(result);
  }
  state.counters["calc"] = benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_V2)->DenseRange(0, 2);

static void BM_Mandelbrot_V3(benchmark::State &state) {
  auto const &test_point = test_points[state.range(0)];
  auto c = test_point.point;
  state.SetLabel(std::format("Local calculation [{}]", test_point.name));
  for (auto _ : state) {
    auto result = mandelbrot::v3::mandelbrot<MAX_ITER>(c);
    benchmark::DoNotOptimize(result);
  }
  state.counters["calc"] = benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_V3)->DenseRange(0, 2);

static void BM_Mandelbrot_V4(benchmark::State &state) {
  auto const &test_point = test_points[state.range(0)];
  auto c = test_point.point;
  state.SetLabel(std::format("Remove std::complex abstraction [{}]", test_point.name));
  for (auto _ : state) {
    auto result = mandelbrot::v4::mandelbrot<MAX_ITER>(c);
    benchmark::DoNotOptimize(result);
  }
  state.counters["calc"] = benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_V4)->DenseRange(0, 2);

static void BM_Mandelbrot_V5(benchmark::State &state) {
  auto const &test_point = test_points[state.range(0)];
  auto c = test_point.point;
  state.SetLabel(std::format("Save partial calculations [{}]", test_point.name));
  for (auto _ : state) {
    auto result = mandelbrot::v5::mandelbrot<MAX_ITER>(c);
    benchmark::DoNotOptimize(result);
  }
  state.counters["calc"] = benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_V5)->DenseRange(0, 2);

static void BM_Mandelbrot_V6(benchmark::State &state) {
  using batch = xsimd::batch<double>;
  constexpr auto width = batch::size;

  auto const &test_point = test_points[state.range(0)];
  auto a = batch(test_point.point.real());
  auto b = batch(test_point.point.imag());
  state.SetLabel(std::format("SIMD [{}]", test_point.name));
  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);
    auto result = mandelbrot::v6::mandelbrot<MAX_ITER>(a, b);
    benchmark::DoNotOptimize(result);
  }
  state.counters["calc"] = benchmark::Counter(width, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_V6)->DenseRange(0, 2);

static void BM_Mandelbrot_V7(benchmark::State &state) {
  using batch = xsimd::batch<double>;
  constexpr auto width = batch::size;

  auto const &test_point = test_points[state.range(0)];
  auto a = batch(test_point.point.real());
  auto b = batch(test_point.point.imag());
  state.SetLabel(std::format("SIMD + unroll + fewer escape [{}]", test_point.name));
  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);
    auto result = mandelbrot::v7::mandelbrot<MAX_ITER>(a, b);
    benchmark::DoNotOptimize(result);
  }
  state.counters["calc"] = benchmark::Counter(width, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_V7)->DenseRange(0, 2);

static void BM_Mandelbrot_MT(benchmark::State &state) {
  auto const &test_point = test_points[state.range(0)];
  state.SetLabel(std::format("Multithreaded [{}]", test_point.name));

  auto c = test_point.point;
  auto gen = [=](std::size_t) { return c; };

  auto scheduler = pool->get_scheduler();
  for (auto _ : state) {
    benchmark::DoNotOptimize(gen);
    auto start = std::chrono::high_resolution_clock::now();
    mandelbrot::v8::mandelbrot<MAX_ITER>(data, gen, scheduler);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    benchmark::ClobberMemory();
  }
  state.counters["calc"] =
      benchmark::Counter(double(state.range(1)), benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_MT)
    ->UseManualTime()
    ->Setup(MTSetup)
    ->Teardown(MTTeardown)
    ->Args({0, PIXEL_COUNT, THREAD_COUNT})
    ->Args({1, PIXEL_COUNT, THREAD_COUNT})
    ->Args({2, PIXEL_COUNT, THREAD_COUNT});

static void BM_Mandelbrot_MT_SIMD(benchmark::State &state) {
  auto const &test_point = test_points[state.range(0)];
  state.SetLabel(std::format("Multithreaded + SIMD [{}]", test_point.name));

  using batch = xsimd::batch<double>;
  auto a = batch(test_point.point.real());
  auto b = batch(test_point.point.imag());
  auto gen = [=](std::size_t) { return std::pair{a, b}; };

  auto scheduler = pool->get_scheduler();
  for (auto _ : state) {
    benchmark::DoNotOptimize(gen);
    auto start = std::chrono::high_resolution_clock::now();
    mandelbrot::v8::mandelbrot<MAX_ITER>(data_simd, gen, scheduler);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    benchmark::ClobberMemory();
  }
  state.counters["calc"] =
      benchmark::Counter(double(state.range(1)), benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Mandelbrot_MT_SIMD)
    ->UseManualTime()
    ->Setup(MTSetup)
    ->Teardown(MTTeardown)
    ->Args({0, PIXEL_COUNT, THREAD_COUNT})
    ->Args({1, PIXEL_COUNT, THREAD_COUNT})
    ->Args({2, PIXEL_COUNT, THREAD_COUNT});

BENCHMARK_MAIN();