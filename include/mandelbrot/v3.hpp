#pragma once

#include <complex>

namespace mandelbrot::v3 {

template <std::size_t MAX_ITER>
[[nodiscard]] auto mandelbrot(std::complex<double> c) -> std::size_t {
  auto iter = std::size_t{};

  auto not_escaped = [](std::complex<double> z) {
    auto x = z.real();
    auto y = z.imag();
    return x * x + y * y <= 4.0;
  };

  auto z = std::complex<double>{};
  while (not_escaped(z) and iter < MAX_ITER) {
    z = z * z + c;
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v3
