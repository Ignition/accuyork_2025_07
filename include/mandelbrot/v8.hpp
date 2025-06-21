#pragma once

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace mandelbrot::v8 {

namespace {
template <std::size_t MAX_ITER>
constexpr auto mandelbrot_simd =
    [](xsimd::batch<double> a,
       xsimd::batch<double> b) -> xsimd::batch<std::size_t> {
  using batch = xsimd::batch<double>;
  using bsize = xsimd::batch<std::size_t>;

  auto const four = batch(4.0);
  auto const two = batch(2.0);
  auto const one = bsize(1);

  auto x = batch(0.0);
  auto y = batch(0.0);
  auto iter = bsize(0);

  for (std::size_t i = 0; i < MAX_ITER; ++i) {
    auto const x2 = x * x;
    auto const y2 = y * y;

    auto const mask = (x2 + y2) <= four;
    if (none(mask)) {
      break;
    }

    auto const xy = x * y;
    auto const mask_i = batch_bool_cast<std::size_t>(mask);

    // Only update where still running
    x = select(mask, x2 - y2 + a, x);
    y = select(mask, fma(two, xy, b), y);
    iter = select(mask_i, iter + one, iter);
  }

  return iter;
};
} // namespace

template <std::size_t MAX_ITER>
auto mandelbrot(auto vec, auto &&gen, auto scheduler) -> auto {

  auto sender = stdexec::bulk_chunked(
      stdexec::schedule(scheduler),
      stdexec::par,
      vec.size(),
      [&](std::size_t begin, std::size_t end) {
        for (auto i = begin; i != end; ++i) {
          vec[i] = std::apply(mandelbrot_simd<MAX_ITER>, gen(i));
        }
      }
  );

  stdexec::sync_wait(std::move(sender));
  return vec;
}

} // namespace mandelbrot::v8
