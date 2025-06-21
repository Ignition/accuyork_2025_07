#pragma once

#include <complex>
#include <tuple>

namespace mandelbrot::v5 {

template <std::size_t MAX_ITER = 10'000>
[[nodiscard]] auto mandelbrot(std::complex<double> c) -> std::size_t {
  auto const a = c.real();
  auto const b = c.imag();

  auto iter = std::size_t{};

  auto x = 0.0;
  auto y = 0.0;
  auto x2 = 0.0;
  auto y2 = 0.0;
  while (x2 + y2 <= 4.0 && iter < MAX_ITER) {
    auto x_next = x2 - y2 + a;
    auto y_next = 2 * x * y + b;
    std::tie(x, y) = std::tie(x_next, y_next);
    y2 = y * y; // store to reuse in the loop check
    x2 = x * x; // store to reuse in the loop check
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v5
