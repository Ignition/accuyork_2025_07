#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "mandelbrot/mandelbrot.hpp"

constexpr auto MAX_LIMIT = 10'000uz;

using sut_t = mandelbrot::v2::mandelbrot<double>;

TEST_CASE("mandelbrot: known bounded point") {

  const auto result =
      sut_t::iteration_count<MAX_LIMIT>(std::complex<double>{0.0, 0.0});
  CHECK(result == MAX_LIMIT); // Should not escape
}

TEST_CASE("mandelbrot: known escaping point") {
  const auto result =sut_t::iteration_count<MAX_LIMIT>(std::complex<double>{2.0, 2.0});
  CHECK(result < 10'000); // Should escape quickly
}

TEST_CASE("mandelbrot: edge of the set") {
  const auto result =sut_t::iteration_count<MAX_LIMIT>(std::complex<double>{-0.75, 0.1});
  CHECK(result > 32); // Should be slow to escape or remain bounded
}

TEST_CASE("mandelbrot: template instantiation with float") {
  using sut_t = mandelbrot::v2::mandelbrot<float>;
  constexpr auto MAX_LIMIT = 5'000uz;
  const auto result =
      sut_t::iteration_count<MAX_LIMIT>(std::complex<float>{0.0, 0.0});
  CHECK(result == MAX_LIMIT);
}