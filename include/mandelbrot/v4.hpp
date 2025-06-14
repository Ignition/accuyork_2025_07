#pragma once

#include <complex>
#include <tuple>

namespace mandelbrot::v4 {

template <typename T = double, std::size_t MAX_ITER = 10'000>
[[nodiscard]] auto mandelbrot(std::complex<T> c) -> std::size_t {
  auto const a = c.real();
  auto const b = c.imag();

  auto iter = std::size_t{};

  auto x = T{};
  auto y = T{};
  while (x * x + y * y <= T(4) && iter < MAX_ITER) {
    auto x_next = x * x - y * y + a;
    auto y_next = 2 * x * y + b;
    std::tie(x, y) = std::tie(x_next, y_next);
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v4
