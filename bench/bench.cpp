#include "mandelbrot/mandelbrot.hpp"
#include <benchmark/benchmark.h>
#include <complex>
#include <format>

#include <exec/libdispatch_queue.hpp>

enum struct QUEUE_TYPE : uint8_t { LIB_DISPATCH, STANDARD };

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
  constexpr auto N = batch::size;

  auto a = batch(point.real());
  auto b = batch(point.imag());

  for (auto _ : state) {
    auto result = mandel_func(a, b);
    benchmark::DoNotOptimize(result);
  }

  state.counters["calc"] =
      benchmark::Counter(N, benchmark::Counter::kIsIterationInvariantRate);
}

template <QUEUE_TYPE queue>
static void BM_Mandelbrot_MT(
    benchmark::State &state,
    auto &&mandel_func,
    const std::complex<double> &point,
    auto &&gen_maker,
    auto &&make_output_vec
) {

  constexpr std::size_t N = 1024 * 8;

  auto gen = gen_maker(point);

  // 64KiB
  //  using batch = xsimd::batch<size_t>;
  //  auto ensure_over_alloc = (N + batch::size - 1) / batch::size *
  //  batch::size;
  auto vec = make_output_vec(N); // std::vector<size_t>(ensure_over_alloc);

  // One common thread pool for the whole benchmark

  if constexpr (QUEUE_TYPE::LIB_DISPATCH == queue) {
    auto dispatch = exec::libdispatch_queue();
    auto scheduler = dispatch.get_scheduler();

    for (auto _ : state) {
      vec = mandel_func(std::move(vec), gen, scheduler);
      benchmark::ClobberMemory();
      benchmark::DoNotOptimize(vec);
    }
  } else {
    auto const core_count = std::max(std::thread::hardware_concurrency(), 4u);
    auto pool = exec::static_thread_pool{core_count};
    auto scheduler = pool.get_scheduler();

    for (auto _ : state) {
      vec = mandel_func(std::move(vec), gen, scheduler);
      benchmark::ClobberMemory();
      benchmark::DoNotOptimize(vec);
    }
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

template <QUEUE_TYPE queue>
void register_mt_benchmarks(
    const char *version_name, auto func, auto gen_maker, auto &&make_output_vec
) {
  for (const auto &point : test_points) {
    benchmark::RegisterBenchmark(
        std::format("BM_Mandelbrot_{}/{}", version_name, point.name),
        [=](benchmark::State &state) {
          BM_Mandelbrot_MT<queue>(
              state,
              func,
              point.point,
              gen_maker,
              make_output_vec
          );
        }
    );
  }
}

// Register all benchmark combinations
int main(int argc, char **argv) {

  constexpr auto MAX_ITER = 10'000uz;
  auto const core_count = std::max(std::thread::hardware_concurrency(), 4u);

  //  register_scalar_benchmarks("V1", [](std::complex<double> const &c) {
  //    return mandelbrot::v1::mandelbrot<MAX_ITER>(c);
  //  });
  //
  //  register_scalar_benchmarks("V2", [](std::complex<double> const &c) {
  //    return mandelbrot::v2::mandelbrot<MAX_ITER>(c);
  //  });
  //
  //  register_scalar_benchmarks("V3", [](std::complex<double> const &c) {
  //    return mandelbrot::v3::mandelbrot<MAX_ITER>(c);
  //  });
  //
  //  register_scalar_benchmarks("V4", [](std::complex<double> const &c) {
  //    return mandelbrot::v4::mandelbrot<MAX_ITER>(c);
  //  });
  //
  //  register_scalar_benchmarks("V5", [](std::complex<double> const &c) {
  //    return mandelbrot::v5::mandelbrot<MAX_ITER>(c);
  //  });

  // SIMD
  register_simd_benchmarks(
      "V6",
      [](xsimd::batch<double> a, xsimd::batch<double> b) {
        return mandelbrot::v6::mandelbrot<MAX_ITER>(a, b);
      }
  );

  // MT
  auto scalar_gen_maker = [](std::complex<double> const &point) {
    return [=](std::size_t) { return point; };
  };
  auto scalar_make_output_vec = [](std::size_t N) {
    return std::vector<std::size_t>(N);
  };

  auto vector_gen_maker = [](std::complex<double> const &point) {
    return [=](std::size_t) {
      using batch = xsimd::batch<double>;

      auto a = batch(point.real());
      auto b = batch(point.imag());
      return std::tuple{a, b};
    };
  };
  auto vector_make_output_vec = [](std::size_t N) {
    using batch = xsimd::batch<size_t>;
    auto const N_plus = (N + batch::size - 1) / batch::size;
    return std::vector<batch>(N_plus);
  };
//  auto vector_assign = [](xsimd::batch<size_t> & lhs, xsimd::batch<size_t> rhs) {
//    lhs = rhs;
//  };

  register_mt_benchmarks<QUEUE_TYPE::STANDARD>(
      "V7-new_pool",
      [core_count](
          auto vec,
          auto &&gen,
          auto /*ignore*/
      ) {
        // New thread pool per benchmark iteration
        auto pool = exec::static_thread_pool{core_count};
        return mandelbrot::v7::mandelbrot<MAX_ITER>(
            std::move(vec),
            gen,
            pool.get_scheduler()
        );
      },
      scalar_gen_maker,scalar_make_output_vec
  );

  register_mt_benchmarks<QUEUE_TYPE::STANDARD>(
      "V7-pool",
      [](auto vec, auto &&gen, auto scheduler) {
        // Reuse the same thread pool
        return mandelbrot::v7::mandelbrot<MAX_ITER>(
            std::move(vec),
            gen,
            scheduler
        );
      },
      scalar_gen_maker,scalar_make_output_vec
  );

  register_mt_benchmarks<QUEUE_TYPE::LIB_DISPATCH>(
      "V7-dispatch",
      [](auto vec, auto &&gen, auto scheduler) {
        // Reuse the same thread pool
        return mandelbrot::v7::mandelbrot<MAX_ITER>(
            std::move(vec),
            gen,
            scheduler
        );
      },
      scalar_gen_maker,scalar_make_output_vec
  );

  register_mt_benchmarks<QUEUE_TYPE::STANDARD>(
      "V8-pool",
      [](auto vec, auto &&gen, auto scheduler) {
        // Reuse the same thread pool
        return mandelbrot::v8::mandelbrot<MAX_ITER>(
            std::move(vec),
            gen,
            scheduler
        );
      },
      vector_gen_maker,vector_make_output_vec
  );

  register_mt_benchmarks<QUEUE_TYPE::LIB_DISPATCH>(
      "V8-dispatch",
      [](auto vec, auto &&gen, auto scheduler) {
        // Reuse the same thread pool
        return mandelbrot::v8::mandelbrot<MAX_ITER>(
            std::move(vec),
            gen,
            scheduler
        );
      },
      vector_gen_maker,vector_make_output_vec
  );

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}