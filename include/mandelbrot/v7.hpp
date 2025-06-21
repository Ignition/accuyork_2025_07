#pragma once

#include "mandelbrot/v5.hpp"

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace mandelbrot::v7 {

template <std::size_t MAX_ITER>
auto mandelbrot(auto vec, auto &&gen, auto scheduler) {

  auto sender = stdexec::bulk_chunked(
      stdexec::schedule(scheduler),
      stdexec::par,
      vec.size(),
      [&](std::size_t begin, std::size_t end) {
        for (auto i = begin; i != end; ++i) {
          vec[i] = mandelbrot::v5::mandelbrot<MAX_ITER>(gen(i));
        }
      }
  );

  stdexec::sync_wait(std::move(sender));
  return vec;
}

} // namespace mandelbrot::v7
