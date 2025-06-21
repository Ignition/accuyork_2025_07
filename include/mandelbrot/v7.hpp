#pragma once

#include "mandelbrot/v5.hpp"

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace mandelbrot::v7 {

template <std::size_t MAX_ITER>
auto mandelbrot(
    std::vector<size_t> vec, auto &&gen, exec::static_thread_pool &pool
) -> std::vector<size_t> {

  auto sender = stdexec::bulk(
      stdexec::schedule(pool.get_scheduler()),
      stdexec::par,
      vec.size(),
      [&](std::size_t i) {
        vec[i] = mandelbrot::v5::mandelbrot<MAX_ITER>(gen(i));
      }
  );

  stdexec::sync_wait(std::move(sender));
  return vec;
}

} // namespace mandelbrot::v7
