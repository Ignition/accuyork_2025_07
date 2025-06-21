#pragma once

#include <complex>

namespace mandelbrot::v1 {

template <std::size_t MAX_ITER>
[[nodiscard]] auto mandelbrot(std::complex<double> c) -> std::size_t {
  auto iter = std::size_t{};

  auto z = std::complex<double>{};
  while (std::abs(z) <= 2.0 && iter < MAX_ITER) {
    z = z * z + c;
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v1
