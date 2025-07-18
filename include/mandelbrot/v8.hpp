#pragma once

#include <stdexec/execution.hpp>
#include <xsimd/xsimd.hpp>

namespace mandelbrot::v8 {

namespace {

template <std::size_t MAX_ITER>
constexpr auto mandelbrot_scalar = [](std::complex<double> c) -> std::size_t {
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
};

template <std::size_t MAX_ITER>
constexpr auto mandelbrot_simd =
    [](xsimd::batch<double> a, xsimd::batch<double> b) -> xsimd::batch<std::size_t> {
  using batch = xsimd::batch<double>;
  using bsize = xsimd::batch<std::size_t>;

  auto const four = batch(4.0);
  auto const two = batch(2.0);
  auto const one = bsize(1);

  auto x = batch(0.0);
  auto y = batch(0.0);
  auto iter = bsize(0);

#pragma clang loop unroll_count(16)
  for (std::size_t i = 0; i < MAX_ITER; ++i) {
    auto const x2 = x * x;
    auto const y2 = y * y;

    auto const mask = (x2 + y2) <= four;
    if (i % 16 == 0 and none(mask)) {
      break;
    }

    auto const xy = x * y;
    auto const mask_i = batch_bool_cast<std::size_t>(mask);

    x = x2 - y2 + a;
    y = fma(two, xy, b);
    // Only update where still running
    iter = select(mask_i, iter + one, iter);
  }

  return iter;
};
} // namespace

template <std::size_t MAX_ITER>
void mandelbrot(auto &vec, auto &&gen, auto scheduler) {

  constexpr bool is_scalar =
      std::is_same_v<typename std::decay_t<decltype(vec)>::value_type, std::size_t>;

  auto sender =
      stdexec::bulk(stdexec::schedule(scheduler), stdexec::par, vec.size(), [&](std::size_t i) {
        if constexpr (is_scalar) {
          vec[i] = mandelbrot_scalar<MAX_ITER>(gen(i));
        } else {
          vec[i] = std::apply(mandelbrot_simd<MAX_ITER>, gen(i));
        }
      });

  stdexec::sync_wait(std::move(sender));
}

} // namespace mandelbrot::v8
