#pragma once

#include <complex>

namespace mandelbrot::v2 {

template <typename T = double, std::size_t MAX_ITER = 10'000>
[[nodiscard]] auto mandelbrot(std::complex<T> c) -> std::size_t {
  auto iter = std::size_t{};

  auto z = std::complex<T>{};
  while (std::norm(z) <= T(4) && iter < MAX_ITER) {
    z = z * z + c;
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v2
