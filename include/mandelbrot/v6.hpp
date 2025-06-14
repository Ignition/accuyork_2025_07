#pragma once

#include <cstddef>
#include <xsimd/xsimd.hpp>

namespace mandelbrot::v6 {

template <std::size_t MAX_ITER = 10'000,typename T = double>
[[nodiscard]] auto mandelbrot(xsimd::batch<T> a, xsimd::batch<T> b)
    -> xsimd::batch<int64_t> {
  using batch = xsimd::batch<T>;
  using bsize = xsimd::batch<int64_t>;

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
    auto const mask_i = batch_bool_cast<int64_t>(mask);

    // Only update where still running
    x = select(mask, x2 - y2 + a, x);
    y = select(mask, fma(two, xy, b), y);
    iter = select(mask_i, iter + one, iter);
  }

  return iter;
}

} // namespace mandelbrot::v6
