#pragma once

#include <complex>

namespace mandelbrot::v1 {

template <std::size_t MAX_ITER = 10'000,typename T = double>
[[nodiscard]] auto mandelbrot(std::complex<T> c) -> std::size_t {
  auto iter = std::size_t{};

  auto z = std::complex<T>{};
  while (std::abs(z) <= T(2) && iter < MAX_ITER) {
    z = z * z + c;
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v1
