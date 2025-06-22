#pragma once

#include <complex>
#include <tuple>

namespace mandelbrot::v4 {

template <std::size_t MAX_ITER>
[[nodiscard]] auto mandelbrot(std::complex<double> c) -> std::size_t {
  auto const a = c.real();
  auto const b = c.imag();

  auto iter = std::size_t{};

  auto x = 0.0;
  auto y = 0.0;
  while (x * x + y * y <= 4.0 and iter < MAX_ITER) {
    auto x_next = x * x - y * y + a;
    auto y_next = 2 * x * y + b;
    std::tie(x, y) = std::tie(x_next, y_next);
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v4
