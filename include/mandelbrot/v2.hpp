#pragma once

#include <complex>

namespace mandelbrot::v2 {

template <std::size_t MAX_ITER>
[[nodiscard]] auto mandelbrot(std::complex<double> c) -> std::size_t {
  auto iter = std::size_t{};

  auto z = std::complex<double>{};
  while (std::norm(z) <= 4.0 && iter < MAX_ITER) {
    z = z * z + c;
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v2
