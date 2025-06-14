#pragma once

#include <complex>

namespace mandelbrot::v3 {

template <std::size_t MAX_ITER = 10'000,typename T = double>
[[nodiscard]] auto mandelbrot(std::complex<T> c) -> std::size_t {
  auto iter = std::size_t{};

  auto not_escaped = [](std::complex<T> z){
    auto x = z.real();
    auto y = z.imag();
    return x*x + y*y <= T(4);
  };

  auto z = std::complex<T>{};
  while (not_escaped(z) && iter < MAX_ITER) {
    z = z * z + c;
    ++iter;
  }
  return iter;
}

} // namespace mandelbrot::v2
