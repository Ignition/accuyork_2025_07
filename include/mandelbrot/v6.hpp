#pragma once

#include <xsimd/xsimd.hpp>

namespace mandelbrot::v6 {

template <std::size_t MAX_ITER>
[[nodiscard]] auto mandelbrot(xsimd::batch<double> a, xsimd::batch<double> b)
    -> xsimd::batch<std::size_t> {
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
}

} // namespace mandelbrot::v6
