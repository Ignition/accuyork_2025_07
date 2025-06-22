#include <SFML/Graphics.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exec/static_thread_pool.hpp>
#include <sstream>
#include <string_view>
#include <vector>
#include <xsimd/xsimd.hpp>

using batch_d = xsimd::batch<double>;
inline constexpr std::size_t MAX_ITER = 1000;

// UI Constants
inline constexpr float LOADING_TEXT_OFFSET = 50.0f;
inline constexpr int DEFAULT_FONT_SIZE = 24;
inline constexpr int HELP_FONT_SIZE = 18;
inline constexpr int TITLE_FONT_SIZE = 24;
inline constexpr float HELP_LINE_SPACING = 26.0f;
inline constexpr float HELP_PANEL_PADDING = 80.0f;
inline constexpr float MIN_SCREEN_MARGIN = 40.0f;

// SIMD Constants  
inline constexpr std::size_t ESCAPE_CHECK_INTERVAL = 16;
inline constexpr double ESCAPE_RADIUS_SQUARED = 4.0;
inline constexpr double SMOOTH_LOG_ESCAPE_RADIUS = 1.3862943611198906; // log(4.0)

// Animation Constants
inline constexpr float SPINNER_ROTATION_INCREMENT = 5.0f;
inline constexpr float MAX_ROTATION_DEGREES = 360.0f;

namespace {

template <typename T>
xsimd::batch<T> iota_batch(T start) {
  using batch_t = xsimd::batch<T>;
  alignas(alignof(batch_t)) T tmp[batch_t::size];
  for (std::size_t i = 0; i != batch_t::size; ++i) {
    tmp[i] = start + static_cast<T>(i);
  }
  return batch_t::load_aligned(tmp);
}

template <std::size_t MAX_ITER>
constexpr auto mandelbrot_simd =
    [](xsimd::batch<double> a,
       xsimd::batch<double> b) -> std::pair<xsimd::batch<std::size_t>, xsimd::batch<double>> {
  using batch = xsimd::batch<double>;
  using bsize = xsimd::batch<std::size_t>;

  auto const four = batch(4.0);
  auto const two = batch(2.0);
  auto const one = bsize(1);

  auto x = batch(0.0);
  auto y = batch(0.0);
  auto iter = bsize(0);

  auto x2 = x * x;
  auto y2 = y * y;
  auto mag = x2 + y2;

#pragma clang loop unroll_count(16)
  for (std::size_t i = 0; i < MAX_ITER; ++i) {

    auto const mask = mag <= batch_d(ESCAPE_RADIUS_SQUARED);
    if (i % ESCAPE_CHECK_INTERVAL == 0 and none(mask)) {
      break;
    }

    auto const xy = x * y;
    auto const mask_i = batch_bool_cast<std::size_t>(mask);

    x = x2 - y2 + a;
    y = fma(two, xy, b);
    x2 = x * x;
    y2 = y * y;
    // Only update where still running
    iter = select(mask_i, iter + one, iter);
    mag = select(mask, x2 + y2, mag);
  }

  return {iter, mag};
};

// ===== UTILITY METHODS =====
[[nodiscard]] constexpr batch_d lerp_simd(const batch_d &a, const batch_d &b, const batch_d &f) noexcept {
  return a + f * (b - a);
}

[[nodiscard]] batch_d labToXyz_simd(const batch_d &t) noexcept {
  static constexpr double DELTA = 6.0 / 29.0;
  static constexpr double DELTA_SQUARED_TIMES_3 = 3.0 * DELTA * DELTA;
  static constexpr double OFFSET = 4.0 / 29.0;
  
  const auto delta = batch_d(DELTA);
  const auto cube = t * t * t;
  const auto linear = batch_d(DELTA_SQUARED_TIMES_3) * (t - batch_d(OFFSET));
  return select(t > delta, cube, linear);
}

[[nodiscard]] batch_d gammaCorrect_simd(const batch_d &c) noexcept {
  static constexpr double LINEAR_FACTOR = 12.92;
  static constexpr double GAMMA_FACTOR = 1.055;
  static constexpr double GAMMA_POWER = 1.0 / 2.4;
  static constexpr double GAMMA_OFFSET = 0.055;
  static constexpr double THRESHOLD = 0.0031308;
  
  const auto linear = batch_d(LINEAR_FACTOR) * c;
  const auto gamma = batch_d(GAMMA_FACTOR) * xsimd::pow(c, batch_d(GAMMA_POWER)) - batch_d(GAMMA_OFFSET);
  return select(c <= batch_d(THRESHOLD), linear, gamma);
}

// ===== COLOUR FUNCTIONS =====

[[nodiscard]] constexpr batch_d clampNormalized(const batch_d &value) noexcept {
  return xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), value));
}

std::tuple<batch_d, batch_d, batch_d> getExponentialLCH_simd(const batch_d &smooth_iterations) {
  // SIMD implementation of Smooth Exponential LCH Color algorithm

  // Handle max iterations (inside set) -> black
  auto max_iter_mask = smooth_iterations >= batch_d(static_cast<double>(MAX_ITER));

  // Calculate s parameter
  auto s = smooth_iterations / batch_d(static_cast<double>(MAX_ITER));

  // Calculate v parameter: v = 1.0 - cos²(π * s)
  auto pi_s = s * batch_d(std::numbers::pi_v<double>);
  auto cos_pi_s = xsimd::cos(pi_s);
  auto v = batch_d(1.0) - cos_pi_s * cos_pi_s;

  // Calculate LCH parameters
  auto L = batch_d(75.0) - (batch_d(75.0) * v);
  auto C = batch_d(28.0) + (batch_d(75.0) - (batch_d(75.0) * v));
  auto H = xsimd::fmod(xsimd::pow(batch_d(360.0) * s, batch_d(1.5)), batch_d(360.0));

  // Convert LCH to LAB
  auto H_rad = H * batch_d(std::numbers::pi_v<double> / 180.0);
  auto lab_a = C * xsimd::cos(H_rad);
  auto lab_b = C * xsimd::sin(H_rad);

  // Convert LAB to XYZ
  auto fy = (L + batch_d(16.0)) / batch_d(116.0);
  auto fx = lab_a / batch_d(500.0) + fy;
  auto fz = fy - lab_b / batch_d(200.0);

  auto X = batch_d(0.95047) * labToXyz_simd(fx);
  auto Y = batch_d(1.00000) * labToXyz_simd(fy);
  auto Z = batch_d(1.08883) * labToXyz_simd(fz);

  // Convert XYZ to linear RGB
  auto R_linear = batch_d(3.2406) * X - batch_d(1.5372) * Y - batch_d(0.4986) * Z;
  auto G_linear = batch_d(-0.9689) * X + batch_d(1.8758) * Y + batch_d(0.0415) * Z;
  auto B_linear = batch_d(0.0557) * X - batch_d(0.2040) * Y + batch_d(1.0570) * Z;

  auto R_srgb = gammaCorrect_simd(R_linear);
  auto G_srgb = gammaCorrect_simd(G_linear);
  auto B_srgb = gammaCorrect_simd(B_linear);

  // Clamp to [0, 1] range
  auto r = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), R_srgb));
  auto g = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), G_srgb));
  auto b = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), B_srgb));

  // Apply black for max iterations
  r = select(max_iter_mask, batch_d(0.0), r);
  g = select(max_iter_mask, batch_d(0.0), g);
  b = select(max_iter_mask, batch_d(0.0), b);

  return {r, g, b};
}

std::tuple<batch_d, batch_d, batch_d> getClassicColor_simd(const batch_d &t) {
  // Classic Ultra Fractal color scheme
  const batch_d t0 = batch_d(0.16);
  const batch_d t1 = batch_d(0.42);
  const batch_d t2 = batch_d(0.6425);
  const batch_d t3 = batch_d(0.8575);

  // Color stops normalized to 0-1
  const batch_d c0_r = batch_d(0.0), c0_g = batch_d(7.0 / 255.0), c0_b = batch_d(100.0 / 255.0);
  const batch_d c1_r = batch_d(32.0 / 255.0), c1_g = batch_d(107.0 / 255.0),
                c1_b = batch_d(203.0 / 255.0);
  const batch_d c2_r = batch_d(237.0 / 255.0), c2_g = batch_d(1.0), c2_b = batch_d(1.0);
  const batch_d c3_r = batch_d(1.0), c3_g = batch_d(170.0 / 255.0), c3_b = batch_d(0.0);
  const batch_d c4_r = batch_d(0.0), c4_g = batch_d(2.0 / 255.0), c4_b = batch_d(0.0);
  const batch_d c5_r = batch_d(0.0), c5_g = batch_d(7.0 / 255.0), c5_b = batch_d(100.0 / 255.0);

  batch_d f01 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), t / t0));
  batch_d f12 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), (t - t0) / (t1 - t0)));
  batch_d f23 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), (t - t1) / (t2 - t1)));
  batch_d f34 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), (t - t2) / (t3 - t2)));
  batch_d f45 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), (t - t3) / (batch_d(1.0) - t3)));

  batch_d r = lerp_simd(c0_r, c1_r, f01);
  batch_d g = lerp_simd(c0_g, c1_g, f01);
  batch_d b = lerp_simd(c0_b, c1_b, f01);

  r = select(t >= t0, lerp_simd(c1_r, c2_r, f12), r);
  g = select(t >= t0, lerp_simd(c1_g, c2_g, f12), g);
  b = select(t >= t0, lerp_simd(c1_b, c2_b, f12), b);

  r = select(t >= t1, lerp_simd(c2_r, c3_r, f23), r);
  g = select(t >= t1, lerp_simd(c2_g, c3_g, f23), g);
  b = select(t >= t1, lerp_simd(c2_b, c3_b, f23), b);

  r = select(t >= t2, lerp_simd(c3_r, c4_r, f34), r);
  g = select(t >= t2, lerp_simd(c3_g, c4_g, f34), g);
  b = select(t >= t2, lerp_simd(c3_b, c4_b, f34), b);

  r = select(t >= t3, lerp_simd(c4_r, c5_r, f45), r);
  g = select(t >= t3, lerp_simd(c4_g, c5_g, f45), g);
  b = select(t >= t3, lerp_simd(c4_b, c5_b, f45), b);

  return {r, g, b};
}

std::tuple<batch_d, batch_d, batch_d> getHotIronColor_simd(const batch_d &t) {
  static constexpr double t0 = 0.25, t1 = 0.5, t2 = 0.75;
  static constexpr double inv_t0 = 4.0; // 1.0 / 0.25
  static constexpr double inv_t1_t0 = 4.0; // 1.0 / (0.5 - 0.25)
  static constexpr double inv_t2_t1 = 4.0; // 1.0 / (0.75 - 0.5)
  static constexpr double inv_1_t2 = 4.0; // 1.0 / (1.0 - 0.75)

  static constexpr double c0_r = 0.0, c0_g = 0.0, c0_b = 0.0;
  static constexpr double c1_r = 0.5, c1_g = 0.0, c1_b = 0.0;
  static constexpr double c2_r = 1.0, c2_g = 0.0, c2_b = 0.0;
  static constexpr double c3_r = 1.0, c3_g = 165.0 / 255.0, c3_b = 0.0;
  static constexpr double c4_r = 1.0, c4_g = 1.0, c4_b = 1.0;

  batch_d f01 = clampNormalized(t * batch_d(inv_t0));
  batch_d f12 = clampNormalized((t - batch_d(t0)) * batch_d(inv_t1_t0));
  batch_d f23 = clampNormalized((t - batch_d(t1)) * batch_d(inv_t2_t1));
  batch_d f34 = clampNormalized((t - batch_d(t2)) * batch_d(inv_1_t2));

  batch_d r = lerp_simd(batch_d(c0_r), batch_d(c1_r), f01);
  batch_d g = lerp_simd(batch_d(c0_g), batch_d(c1_g), f01);
  batch_d b = lerp_simd(batch_d(c0_b), batch_d(c1_b), f01);

  r = select(t >= batch_d(t0), lerp_simd(batch_d(c1_r), batch_d(c2_r), f12), r);
  g = select(t >= batch_d(t0), lerp_simd(batch_d(c1_g), batch_d(c2_g), f12), g);
  b = select(t >= batch_d(t0), lerp_simd(batch_d(c1_b), batch_d(c2_b), f12), b);

  r = select(t >= batch_d(t1), lerp_simd(batch_d(c2_r), batch_d(c3_r), f23), r);
  g = select(t >= batch_d(t1), lerp_simd(batch_d(c2_g), batch_d(c3_g), f23), g);
  b = select(t >= batch_d(t1), lerp_simd(batch_d(c2_b), batch_d(c3_b), f23), b);

  r = select(t >= batch_d(t2), lerp_simd(batch_d(c3_r), batch_d(c4_r), f34), r);
  g = select(t >= batch_d(t2), lerp_simd(batch_d(c3_g), batch_d(c4_g), f34), g);
  b = select(t >= batch_d(t2), lerp_simd(batch_d(c3_b), batch_d(c4_b), f34), b);

  return {r, g, b};
}

std::tuple<batch_d, batch_d, batch_d> getElectricBlueColor_simd(const batch_d &t) {
  const batch_d c0_r = batch_d(0.0), c0_g = batch_d(0.0), c0_b = batch_d(50.0 / 255.0);
  const batch_d c1_r = batch_d(0.0), c1_g = batch_d(100.0 / 255.0), c1_b = batch_d(1.0);
  const batch_d c2_r = batch_d(0.0), c2_g = batch_d(1.0), c2_b = batch_d(1.0);

  auto mask1 = t < batch_d(0.5);
  auto f1 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), t / batch_d(0.5)));
  auto f2 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), (t - batch_d(0.5)) / batch_d(0.5)));

  auto r = select(mask1, lerp_simd(c0_r, c1_r, f1), lerp_simd(c1_r, c2_r, f2));
  auto g = select(mask1, lerp_simd(c0_g, c1_g, f1), lerp_simd(c1_g, c2_g, f2));
  auto b = select(mask1, lerp_simd(c0_b, c1_b, f1), lerp_simd(c1_b, c2_b, f2));

  return {r, g, b};
}

std::tuple<batch_d, batch_d, batch_d> getSunsetColor_simd(const batch_d &t) {
  const batch_d t0 = batch_d(0.33);
  const batch_d t1 = batch_d(0.66);

  const batch_d c0_r = batch_d(25.0 / 255.0), c0_g = batch_d(0.0), c0_b = batch_d(51.0 / 255.0);
  const batch_d c1_r = batch_d(1.0), c1_g = batch_d(0.0), c1_b = batch_d(127.0 / 255.0);
  const batch_d c2_r = batch_d(1.0), c2_g = batch_d(127.0 / 255.0), c2_b = batch_d(0.0);
  const batch_d c3_r = batch_d(1.0), c3_g = batch_d(1.0), c3_b = batch_d(0.0);

  batch_d f01 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), t / t0));
  batch_d f12 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), (t - t0) / (t1 - t0)));
  batch_d f23 = xsimd::min(batch_d(1.0), xsimd::max(batch_d(0.0), (t - t1) / (batch_d(1.0) - t1)));

  batch_d r = lerp_simd(c0_r, c1_r, f01);
  batch_d g = lerp_simd(c0_g, c1_g, f01);
  batch_d b = lerp_simd(c0_b, c1_b, f01);

  r = select(t >= t0, lerp_simd(c1_r, c2_r, f12), r);
  g = select(t >= t0, lerp_simd(c1_g, c2_g, f12), g);
  b = select(t >= t0, lerp_simd(c1_b, c2_b, f12), b);

  r = select(t >= t1, lerp_simd(c2_r, c3_r, f23), r);
  g = select(t >= t1, lerp_simd(c2_g, c3_g, f23), g);
  b = select(t >= t1, lerp_simd(c2_b, c3_b, f23), b);

  return {r, g, b};
}

[[nodiscard]] constexpr std::tuple<batch_d, batch_d, batch_d> getGrayscaleColor_simd(const batch_d &t) noexcept { 
  return {t, t, t}; 
}

std::tuple<batch_d, batch_d, batch_d> getBlueWhiteColor_simd(const batch_d &t) {
  static constexpr double c0_r = 0.0, c0_g = 50.0 / 255.0, c0_b = 150.0 / 255.0;
  static constexpr double c1_r = 1.0, c1_g = 1.0, c1_b = 1.0;

  return {
    lerp_simd(batch_d(c0_r), batch_d(c1_r), t), 
    lerp_simd(batch_d(c0_g), batch_d(c1_g), t), 
    lerp_simd(batch_d(c0_b), batch_d(c1_b), t)
  };
}

// 🌈 Rainbow Spiral - Smooth HSV rainbow with spiral effect
std::tuple<batch_d, batch_d, batch_d> getRainbowSpiralColor_simd(const batch_d &t) {
  // Create spiral effect with frequency modulation
  auto spiral_t = xsimd::fmod(t * batch_d(3.0), batch_d(1.0));
  
  // Convert to HSV where H cycles through rainbow
  auto hue = spiral_t * batch_d(360.0); // Full rainbow cycle
  auto sat = batch_d(0.85) + batch_d(0.15) * xsimd::sin(t * batch_d(8.0)); // Slight saturation variation
  auto val = batch_d(0.9) + batch_d(0.1) * xsimd::cos(t * batch_d(12.0)); // Slight brightness variation
  
  // Simple HSV to RGB conversion for hue cycling
  auto h_norm = xsimd::fmod(hue / batch_d(60.0), batch_d(6.0));
  auto chroma = val * sat;
  auto x = chroma * (batch_d(1.0) - xsimd::abs(xsimd::fmod(h_norm, batch_d(2.0)) - batch_d(1.0)));
  auto m = val - chroma;
  
  // Determine RGB based on hue sector
  auto mask0 = h_norm < batch_d(1.0);
  auto mask1 = (h_norm >= batch_d(1.0)) & (h_norm < batch_d(2.0));
  auto mask2 = (h_norm >= batch_d(2.0)) & (h_norm < batch_d(3.0));
  auto mask3 = (h_norm >= batch_d(3.0)) & (h_norm < batch_d(4.0));
  auto mask4 = (h_norm >= batch_d(4.0)) & (h_norm < batch_d(5.0));
  
  auto r = select(mask0, chroma, select(mask1, x, select(mask2, batch_d(0.0), select(mask3, batch_d(0.0), select(mask4, x, chroma))))) + m;
  auto g = select(mask0, x, select(mask1, chroma, select(mask2, chroma, select(mask3, x, select(mask4, batch_d(0.0), batch_d(0.0)))))) + m;
  auto b = select(mask0, batch_d(0.0), select(mask1, batch_d(0.0), select(mask2, x, select(mask3, chroma, select(mask4, chroma, x))))) + m;
  
  return {r, g, b};
}

// 🌊 Ocean Depths - Deep blues to aqua to white foam
std::tuple<batch_d, batch_d, batch_d> getOceanDepthsColor_simd(const batch_d &t) {
  static constexpr double t0 = 0.3, t1 = 0.6, t2 = 0.85;
  
  // Deep ocean blue → Turquoise → Aqua → White foam
  static constexpr double c0_r = 0.0, c0_g = 0.1, c0_b = 0.3;      // Deep blue
  static constexpr double c1_r = 0.0, c1_g = 0.4, c1_b = 0.7;      // Medium blue
  static constexpr double c2_r = 0.0, c2_g = 0.8, c2_b = 0.9;      // Turquoise  
  static constexpr double c3_r = 0.7, c3_g = 1.0, c3_b = 1.0;      // Light aqua
  static constexpr double c4_r = 1.0, c4_g = 1.0, c4_b = 1.0;      // White foam
  
  auto f01 = clampNormalized(t / batch_d(t0));
  auto f12 = clampNormalized((t - batch_d(t0)) / batch_d(t1 - t0));
  auto f23 = clampNormalized((t - batch_d(t1)) / batch_d(t2 - t1));
  auto f34 = clampNormalized((t - batch_d(t2)) / batch_d(1.0 - t2));
  
  auto r = lerp_simd(batch_d(c0_r), batch_d(c1_r), f01);
  auto g = lerp_simd(batch_d(c0_g), batch_d(c1_g), f01);
  auto b = lerp_simd(batch_d(c0_b), batch_d(c1_b), f01);
  
  r = select(t >= batch_d(t0), lerp_simd(batch_d(c1_r), batch_d(c2_r), f12), r);
  g = select(t >= batch_d(t0), lerp_simd(batch_d(c1_g), batch_d(c2_g), f12), g);
  b = select(t >= batch_d(t0), lerp_simd(batch_d(c1_b), batch_d(c2_b), f12), b);
  
  r = select(t >= batch_d(t1), lerp_simd(batch_d(c2_r), batch_d(c3_r), f23), r);
  g = select(t >= batch_d(t1), lerp_simd(batch_d(c2_g), batch_d(c3_g), f23), g);
  b = select(t >= batch_d(t1), lerp_simd(batch_d(c2_b), batch_d(c3_b), f23), b);
  
  r = select(t >= batch_d(t2), lerp_simd(batch_d(c3_r), batch_d(c4_r), f34), r);
  g = select(t >= batch_d(t2), lerp_simd(batch_d(c3_g), batch_d(c4_g), f34), g);
  b = select(t >= batch_d(t2), lerp_simd(batch_d(c3_b), batch_d(c4_b), f34), b);
  
  return {r, g, b};
}

// 🔥 Lava Flow - Black → deep red → orange → yellow → white
std::tuple<batch_d, batch_d, batch_d> getLavaFlowColor_simd(const batch_d &t) {
  static constexpr double t0 = 0.2, t1 = 0.4, t2 = 0.7, t3 = 0.9;
  
  // Volcanic progression
  static constexpr double c0_r = 0.05, c0_g = 0.0, c0_b = 0.0;     // Nearly black
  static constexpr double c1_r = 0.4, c1_g = 0.0, c1_b = 0.0;      // Deep red
  static constexpr double c2_r = 0.8, c2_g = 0.2, c2_b = 0.0;      // Orange-red
  static constexpr double c3_r = 1.0, c3_g = 0.6, c3_b = 0.0;      // Orange
  static constexpr double c4_r = 1.0, c4_g = 1.0, c4_b = 0.4;      // Yellow
  static constexpr double c5_r = 1.0, c5_g = 1.0, c5_b = 1.0;      // White hot
  
  auto f01 = clampNormalized(t / batch_d(t0));
  auto f12 = clampNormalized((t - batch_d(t0)) / batch_d(t1 - t0));
  auto f23 = clampNormalized((t - batch_d(t1)) / batch_d(t2 - t1));
  auto f34 = clampNormalized((t - batch_d(t2)) / batch_d(t3 - t2));
  auto f45 = clampNormalized((t - batch_d(t3)) / batch_d(1.0 - t3));
  
  auto r = lerp_simd(batch_d(c0_r), batch_d(c1_r), f01);
  auto g = lerp_simd(batch_d(c0_g), batch_d(c1_g), f01);
  auto b = lerp_simd(batch_d(c0_b), batch_d(c1_b), f01);
  
  r = select(t >= batch_d(t0), lerp_simd(batch_d(c1_r), batch_d(c2_r), f12), r);
  g = select(t >= batch_d(t0), lerp_simd(batch_d(c1_g), batch_d(c2_g), f12), g);
  b = select(t >= batch_d(t0), lerp_simd(batch_d(c1_b), batch_d(c2_b), f12), b);
  
  r = select(t >= batch_d(t1), lerp_simd(batch_d(c2_r), batch_d(c3_r), f23), r);
  g = select(t >= batch_d(t1), lerp_simd(batch_d(c2_g), batch_d(c3_g), f23), g);
  b = select(t >= batch_d(t1), lerp_simd(batch_d(c2_b), batch_d(c3_b), f23), b);
  
  r = select(t >= batch_d(t2), lerp_simd(batch_d(c3_r), batch_d(c4_r), f34), r);
  g = select(t >= batch_d(t2), lerp_simd(batch_d(c3_g), batch_d(c4_g), f34), g);
  b = select(t >= batch_d(t2), lerp_simd(batch_d(c3_b), batch_d(c4_b), f34), b);
  
  r = select(t >= batch_d(t3), lerp_simd(batch_d(c4_r), batch_d(c5_r), f45), r);
  g = select(t >= batch_d(t3), lerp_simd(batch_d(c4_g), batch_d(c5_g), f45), g);
  b = select(t >= batch_d(t3), lerp_simd(batch_d(c4_b), batch_d(c5_b), f45), b);
  
  return {r, g, b};
}

// 🌸 Cherry Blossom - Soft pinks and whites with touches of green
std::tuple<batch_d, batch_d, batch_d> getCherryBlossomColor_simd(const batch_d &t) {
  static constexpr double t0 = 0.25, t1 = 0.5, t2 = 0.75;
  
  // Delicate spring colors
  static constexpr double c0_r = 0.2, c0_g = 0.4, c0_b = 0.2;      // Soft green
  static constexpr double c1_r = 0.9, c1_g = 0.7, c1_b = 0.8;      // Light pink
  static constexpr double c2_r = 1.0, c2_g = 0.8, c2_b = 0.9;      // Pale pink
  static constexpr double c3_r = 0.95, c3_g = 0.5, c3_b = 0.7;     // Cherry blossom pink
  static constexpr double c4_r = 1.0, c4_g = 1.0, c4_b = 1.0;      // Pure white
  
  auto f01 = clampNormalized(t / batch_d(t0));
  auto f12 = clampNormalized((t - batch_d(t0)) / batch_d(t1 - t0));
  auto f23 = clampNormalized((t - batch_d(t1)) / batch_d(t2 - t1));
  auto f34 = clampNormalized((t - batch_d(t2)) / batch_d(1.0 - t2));
  
  auto r = lerp_simd(batch_d(c0_r), batch_d(c1_r), f01);
  auto g = lerp_simd(batch_d(c0_g), batch_d(c1_g), f01);
  auto b = lerp_simd(batch_d(c0_b), batch_d(c1_b), f01);
  
  r = select(t >= batch_d(t0), lerp_simd(batch_d(c1_r), batch_d(c2_r), f12), r);
  g = select(t >= batch_d(t0), lerp_simd(batch_d(c1_g), batch_d(c2_g), f12), g);
  b = select(t >= batch_d(t0), lerp_simd(batch_d(c1_b), batch_d(c2_b), f12), b);
  
  r = select(t >= batch_d(t1), lerp_simd(batch_d(c2_r), batch_d(c3_r), f23), r);
  g = select(t >= batch_d(t1), lerp_simd(batch_d(c2_g), batch_d(c3_g), f23), g);
  b = select(t >= batch_d(t1), lerp_simd(batch_d(c2_b), batch_d(c3_b), f23), b);
  
  r = select(t >= batch_d(t2), lerp_simd(batch_d(c3_r), batch_d(c4_r), f34), r);
  g = select(t >= batch_d(t2), lerp_simd(batch_d(c3_g), batch_d(c4_g), f34), g);
  b = select(t >= batch_d(t2), lerp_simd(batch_d(c3_b), batch_d(c4_b), f34), b);
  
  return {r, g, b};
}

// ⚡ Neon Cyberpunk - Electric purple/blue/cyan for futuristic vibes  
std::tuple<batch_d, batch_d, batch_d> getNeonCyberpunkColor_simd(const batch_d &t) {
  static constexpr double t0 = 0.3, t1 = 0.6;
  
  // Cyberpunk neon colors
  static constexpr double c0_r = 0.1, c0_g = 0.0, c0_b = 0.2;      // Dark purple
  static constexpr double c1_r = 0.5, c1_g = 0.0, c1_b = 1.0;      // Electric purple
  static constexpr double c2_r = 0.0, c2_g = 0.5, c2_b = 1.0;      // Electric blue
  static constexpr double c3_r = 0.0, c3_g = 1.0, c3_b = 1.0;      // Cyan
  static constexpr double c4_r = 1.0, c4_g = 1.0, c4_b = 1.0;      // White glow
  
  auto f01 = clampNormalized(t / batch_d(t0));
  auto f12 = clampNormalized((t - batch_d(t0)) / batch_d(t1 - t0));
  auto f23 = clampNormalized((t - batch_d(t1)) / batch_d(1.0 - t1));
  
  auto r = lerp_simd(batch_d(c0_r), batch_d(c1_r), f01);
  auto g = lerp_simd(batch_d(c0_g), batch_d(c1_g), f01);
  auto b = lerp_simd(batch_d(c0_b), batch_d(c1_b), f01);
  
  r = select(t >= batch_d(t0), lerp_simd(batch_d(c1_r), batch_d(c2_r), f12), r);
  g = select(t >= batch_d(t0), lerp_simd(batch_d(c1_g), batch_d(c2_g), f12), g);
  b = select(t >= batch_d(t0), lerp_simd(batch_d(c1_b), batch_d(c2_b), f12), b);
  
  r = select(t >= batch_d(t1), lerp_simd(batch_d(c2_r), batch_d(c4_r), f23), r);
  g = select(t >= batch_d(t1), lerp_simd(batch_d(c2_g), batch_d(c4_g), f23), g);
  b = select(t >= batch_d(t1), lerp_simd(batch_d(c2_b), batch_d(c4_b), f23), b);
  
  return {r, g, b};
}

// 🍂 Autumn Forest - Rich browns, oranges, golds, and deep reds
std::tuple<batch_d, batch_d, batch_d> getAutumnForestColor_simd(const batch_d &t) {
  static constexpr double t0 = 0.2, t1 = 0.4, t2 = 0.7;
  
  // Autumn foliage colors
  static constexpr double c0_r = 0.2, c0_g = 0.1, c0_b = 0.05;     // Dark brown
  static constexpr double c1_r = 0.6, c1_g = 0.3, c1_b = 0.1;      // Rich brown
  static constexpr double c2_r = 0.8, c2_g = 0.4, c2_b = 0.1;      // Orange-brown
  static constexpr double c3_r = 1.0, c3_g = 0.6, c3_b = 0.0;      // Golden orange
  static constexpr double c4_r = 0.8, c4_g = 0.2, c4_b = 0.1;      // Deep red
  static constexpr double c5_r = 1.0, c5_g = 0.8, c5_b = 0.4;      // Golden yellow
  
  auto f01 = clampNormalized(t / batch_d(t0));
  auto f12 = clampNormalized((t - batch_d(t0)) / batch_d(t1 - t0));
  auto f23 = clampNormalized((t - batch_d(t1)) / batch_d(t2 - t1));
  auto f34 = clampNormalized((t - batch_d(t2)) / batch_d(1.0 - t2));
  
  auto r = lerp_simd(batch_d(c0_r), batch_d(c1_r), f01);
  auto g = lerp_simd(batch_d(c0_g), batch_d(c1_g), f01);
  auto b = lerp_simd(batch_d(c0_b), batch_d(c1_b), f01);
  
  r = select(t >= batch_d(t0), lerp_simd(batch_d(c1_r), batch_d(c2_r), f12), r);
  g = select(t >= batch_d(t0), lerp_simd(batch_d(c1_g), batch_d(c2_g), f12), g);
  b = select(t >= batch_d(t0), lerp_simd(batch_d(c1_b), batch_d(c2_b), f12), b);
  
  r = select(t >= batch_d(t1), lerp_simd(batch_d(c2_r), batch_d(c3_r), f23), r);
  g = select(t >= batch_d(t1), lerp_simd(batch_d(c2_g), batch_d(c3_g), f23), g);
  b = select(t >= batch_d(t1), lerp_simd(batch_d(c2_b), batch_d(c3_b), f23), b);
  
  r = select(t >= batch_d(t2), lerp_simd(batch_d(c3_r), batch_d(c5_r), f34), r);
  g = select(t >= batch_d(t2), lerp_simd(batch_d(c3_g), batch_d(c5_g), f34), g);
  b = select(t >= batch_d(t2), lerp_simd(batch_d(c3_b), batch_d(c5_b), f34), b);
  
  return {r, g, b};
}

} // namespace

class MandelbrotViewer {
private:
  // ===== CONSTANTS =====
  static constexpr std::size_t DEFAULT_WIDTH = 800;
  static constexpr std::size_t DEFAULT_HEIGHT = 600;
  static constexpr double DEFAULT_CENTER_X = -0.7;
  static constexpr double DEFAULT_CENTER_Y = 0.0;
  static constexpr double DEFAULT_ZOOM = 0.8;
  static constexpr auto RENDER_DELAY = std::chrono::milliseconds(150);

  // Rendering constants
  static constexpr int MAX_AA_SAMPLES = 4;
  static constexpr double ZOOM_IN_FACTOR = 1.25;
  static constexpr double ZOOM_OUT_FACTOR = 0.8;
  static constexpr double VIEWPORT_SCALE = 3.0;

  // ===== ENUMS =====
  enum class ColorScheme : int {
    CLASSIC = 0,
    HOT_IRON,
    ELECTRIC_BLUE,
    SUNSET,
    GRAYSCALE,
    BLUE_WHITE,
    EXPONENTIAL_LCH,
    RAINBOW_SPIRAL,
    OCEAN_DEPTHS,
    LAVA_FLOW,
    CHERRY_BLOSSOM,
    NEON_CYBERPUNK,
    AUTUMN_FOREST,
    COUNT
  };

  enum class AntiAliasingLevel : int { X1 = 1, X4 = 2, X9 = 3, X16 = 4 };

  [[nodiscard]] static constexpr int toSamples(AntiAliasingLevel level) noexcept {
    switch (level) {
    case AntiAliasingLevel::X1: return 1;
    case AntiAliasingLevel::X4: return 4;
    case AntiAliasingLevel::X9: return 9;
    case AntiAliasingLevel::X16: return 16;
    }
    return 1; // Default fallback
  }

  // ===== GRAPHICS COMPONENTS =====
  sf::RenderWindow window;
  sf::Image image;
  sf::Texture texture;
  sf::Sprite sprite;

  // ===== COMPUTATION =====
  std::unique_ptr<exec::static_thread_pool> thread_pool;

  // ===== VIEWPORT STATE =====
  double center_x = DEFAULT_CENTER_X;
  double center_y = DEFAULT_CENTER_Y;
  double zoom = DEFAULT_ZOOM;
  std::size_t current_width = DEFAULT_WIDTH;
  std::size_t current_height = DEFAULT_HEIGHT;

  // ===== RENDERING OPTIONS =====
  ColorScheme current_color_scheme = ColorScheme::CLASSIC;
  bool anti_aliasing_enabled = false;
  bool smooth_coloring_enabled = false;
  AntiAliasingLevel aa_level = AntiAliasingLevel::X1;

  // ===== INTERACTION STATE =====
  bool is_dragging = false;
  bool is_rendering = false;
  bool is_panning = false;
  sf::Vector2i last_mouse_pos;
  std::chrono::steady_clock::time_point last_pan_time;

  // ===== UI ELEMENTS =====
  sf::Font font;
  sf::Font monospace_font;
  sf::Text loading_text;
  bool show_help = false;
  std::vector<sf::Text> help_texts;

public:
  MandelbrotViewer()
      : window(sf::VideoMode(DEFAULT_WIDTH, DEFAULT_HEIGHT), "Mandelbrot Viewer"),
        thread_pool(
            std::make_unique<exec::static_thread_pool>(std::thread::hardware_concurrency())
        ) {
    initializeGraphics();
    setupUI();
    render();
  }

  void run() {
    while (window.isOpen()) {
      handleEvents();
      checkDelayedRender();
      draw();
    }
  }

private:
  // ===== INITIALIZATION =====
  void initializeGraphics() {
    image.create(current_width, current_height);
    texture.create(current_width, current_height);
    sprite.setTexture(texture);
  }

  void setupUI() {
    // Try to load a system font
    bool font_loaded = false;
    static constexpr std::array<std::string_view, 5> font_paths = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arial.ttf"
    };

    for (const auto path : font_paths) {
      if (font.loadFromFile(std::string{path})) {
        font_loaded = true;
        break;
      }
    }

    // Try to load a monospace font for help text
    bool monospace_font_loaded = false;
    static constexpr std::array<std::string_view, 8> monospace_font_paths = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/consola.ttf"
    };

    for (const auto path : monospace_font_paths) {
      if (monospace_font.loadFromFile(std::string{path})) {
        monospace_font_loaded = true;
        break;
      }
    }

    loading_text.setString("Rendering...");
    loading_text.setCharacterSize(DEFAULT_FONT_SIZE);
    loading_text.setFillColor(sf::Color::White);
    loading_text.setPosition(current_width / 2.0f - LOADING_TEXT_OFFSET, current_height / 2.0f);
    if (font_loaded) {
      loading_text.setFont(font);
    }

    setupHelpTexts(font_loaded, monospace_font_loaded);
  }

  void setupHelpTexts(bool font_loaded, bool monospace_font_loaded) {
    // Create help text content
    static constexpr std::array<std::string_view, 30> help_content = {
        "MANDELBROT VIEWER - CONTROLS",
        "",
        "Navigation:",
        "  Mouse Wheel      - Zoom in/out",
        "  Left Click+Drag  - Pan view",
        "  R                - Reset view",
        "",
        "Rendering:",
        "  S                - Toggle smooth coloring on/off",
        "  A                - Toggle anti-aliasing",
        "  Q                - Cycle anti-aliasing quality",
        "",
        "Color Schemes:",
        "  C                - Cycle color schemes",
        "  1-9, 0           - Direct color scheme selection:",
        "    1 - Ultra Fractal Classic",
        "    2 - Hot Iron",
        "    3 - Electric Blue",
        "    4 - Sunset",
        "    5 - Grayscale",
        "    6 - Blue to White",
        "    7 - Exponential LCH",
        "    8 - Rainbow Spiral",
        "    9 - Ocean Depths",
        "    0 - Lava Flow",
        "",
        "Help:",
        "  H or F1          - Toggle this help",
        "",
        "Press H or F1 to close this help"
    };

    help_texts.clear();
    help_texts.reserve(help_content.size());

    for (std::size_t i = 0; i < help_content.size(); ++i) {
      sf::Text text;
      text.setString(std::string{help_content[i]});
      text.setCharacterSize(HELP_FONT_SIZE);
      text.setFillColor(sf::Color::White);

      // Special formatting for title - use regular font
      if (i == 0) {
        text.setCharacterSize(TITLE_FONT_SIZE);
        if (font_loaded) {
          text.setFont(font);
          text.setStyle(sf::Text::Bold);
        }
      }
      // Special formatting for section headers - use regular font  
      else if (help_content[i].find(':') != std::string_view::npos && help_content[i].substr(0, 2) != "  ") {
        if (font_loaded) {
          text.setFont(font);
          text.setStyle(sf::Text::Bold);
        }
        text.setFillColor(sf::Color::Yellow);
      }
      // Use monospace font for control listings and other content
      else {
        if (monospace_font_loaded) {
          text.setFont(monospace_font);
        } else if (font_loaded) {
          text.setFont(font);
        }
      }

      const float y_pos = 30.0f + static_cast<float>(i) * HELP_LINE_SPACING;
      text.setPosition(30.0f, y_pos);

      help_texts.push_back(text);
    }
  }

  // ===== EVENT HANDLING =====
  void handleEvents() {
    sf::Event event{};
    while (window.pollEvent(event)) {
      switch (event.type) {
      case sf::Event::Closed:
        window.close();
        break;
      case sf::Event::MouseWheelScrolled:
        handleZoom(
            event.mouseWheelScroll.delta,
            event.mouseWheelScroll.x,
            event.mouseWheelScroll.y
        );
        break;
      case sf::Event::MouseButtonPressed:
        if (event.mouseButton.button == sf::Mouse::Left) {
          startDragging(event.mouseButton.x, event.mouseButton.y);
        }
        break;
      case sf::Event::MouseButtonReleased:
        if (event.mouseButton.button == sf::Mouse::Left) {
          stopDragging();
        }
        break;
      case sf::Event::MouseMoved:
        if (is_dragging) {
          handlePan(event.mouseMove.x - last_mouse_pos.x, event.mouseMove.y - last_mouse_pos.y);
          last_mouse_pos = {event.mouseMove.x, event.mouseMove.y};
        }
        break;
      case sf::Event::Resized:
        handleResize(event.size.width, event.size.height);
        break;
      case sf::Event::KeyPressed:
        handleKeyPress(event.key.code);
        break;
      default:
        break;
      }
    }
  }

  void startDragging(int x, int y) {
    is_dragging = true;
    last_mouse_pos = {x, y};
  }

  void stopDragging() {
    is_dragging = false;
    if (is_panning) {
      is_panning = false;
      render();
    }
  }

  void handleKeyPress(sf::Keyboard::Key key) {
    if (key >= sf::Keyboard::Num1 && key <= sf::Keyboard::Num9) {
      const auto scheme_index = key - sf::Keyboard::Num1;
      const auto max_scheme = static_cast<int>(ColorScheme::COUNT);
      if (scheme_index >= 0 && scheme_index < max_scheme) {
        current_color_scheme = static_cast<ColorScheme>(scheme_index);
        render();
      }
    } else if (key == sf::Keyboard::Num0) {
      // Key 0 maps to scheme index 9 (10th scheme)
      const auto scheme_index = 9;
      const auto max_scheme = static_cast<int>(ColorScheme::COUNT);
      if (scheme_index < max_scheme) {
        current_color_scheme = static_cast<ColorScheme>(scheme_index);
        render();
      }
    } else {
      switch (key) {
      case sf::Keyboard::R:
        resetView();
        break;
      case sf::Keyboard::A:
        toggleAntiAliasing();
        break;
      case sf::Keyboard::Q:
        cycleAntiAliasingLevel();
        break;
      case sf::Keyboard::C:
        cycleColorScheme();
        break;
      case sf::Keyboard::S:
        toggleSmoothColoring();
        break;
      case sf::Keyboard::H:
        [[fallthrough]];
      case sf::Keyboard::F1:
        toggleHelp();
        break;
      default:
        break;
      }
    }
  }

  // ===== NAVIGATION =====
  void handleZoom(float delta, int mouse_x, int mouse_y) {
    auto screenToComplex = [&](int screen_x, int screen_y) -> std::pair<double, double> {
      const double scale = VIEWPORT_SCALE / (zoom * std::min(current_width, current_height));
      const double real = center_x + (screen_x - current_width / 2.0) * scale;
      const double imag = center_y - (screen_y - current_height / 2.0) * scale;
      return {real, imag};
    };
    auto [old_real, old_imag] = screenToComplex(mouse_x, mouse_y);
    zoom *= (delta > 0) ? ZOOM_IN_FACTOR : ZOOM_OUT_FACTOR;
    auto [new_real, new_imag] = screenToComplex(mouse_x, mouse_y);
    center_x += old_real - new_real;
    center_y += old_imag - new_imag;
    render();
  }

  void handlePan(int dx, int dy) {
    double scale = VIEWPORT_SCALE / (zoom * std::min(current_width, current_height));
    center_x -= dx * scale;
    center_y += dy * scale;
    is_panning = true;
    last_pan_time = std::chrono::steady_clock::now();
  }

  void handleResize(unsigned int new_width, unsigned int new_height) {
    current_width = new_width;
    current_height = new_height;

    sf::FloatRect visibleArea(0, 0, new_width, new_height);
    window.setView(sf::View(visibleArea));

    image.create(current_width, current_height);
    texture.create(current_width, current_height);
    sprite.setTexture(texture, true);
    sprite.setPosition(0, 0);
    sprite.setScale(1.0f, 1.0f);

    // Update loading text position
    loading_text.setPosition(current_width / 2.0f - 50, current_height / 2.0f);

    render();
  }

  void resetView() {
    center_x = DEFAULT_CENTER_X;
    center_y = DEFAULT_CENTER_Y;
    zoom = DEFAULT_ZOOM;
    render();
  }

  // ===== SETTINGS =====
  void toggleAntiAliasing() {
    anti_aliasing_enabled = !anti_aliasing_enabled;
    render();
  }

  void cycleAntiAliasingLevel() {
    int current_level = static_cast<int>(aa_level);
    int next_level = current_level + 1;
    if (next_level > static_cast<int>(AntiAliasingLevel::X16)) {
      next_level = static_cast<int>(AntiAliasingLevel::X1);
    }
    aa_level = static_cast<AntiAliasingLevel>(next_level);
    render();
  }

  void cycleColorScheme() {
    int next_scheme =
        (static_cast<int>(current_color_scheme) + 1) % static_cast<int>(ColorScheme::COUNT);
    current_color_scheme = static_cast<ColorScheme>(next_scheme);
    render();
  }

  void toggleSmoothColoring() {
    smooth_coloring_enabled = !smooth_coloring_enabled;
    render();
  }

  void toggleHelp() { show_help = !show_help; }

  // ===== RENDERING =====
  void checkDelayedRender() {
    if (is_panning) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_pan_time);
      if (elapsed >= RENDER_DELAY) {
        is_panning = false;
        render();
      }
    }
  }

  void render() {
    is_rendering = true;
    showLoadingIndicator();

    auto start_time = std::chrono::high_resolution_clock::now();

    int samples_per_side = anti_aliasing_enabled ? static_cast<int>(aa_level) : 1;

    renderUnified(samples_per_side);

    texture.update(image);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    is_rendering = false;
    updateWindowTitle(duration.count());
  }

  void renderUnified(int samples_per_side) {
    // Dispatch to template specializations for optimal performance
    auto dispatch_1 = [&]<int SamplesPerSide>() {
      switch (current_color_scheme) {
      case ColorScheme::CLASSIC:
        renderWithSampling<SamplesPerSide, ColorScheme::CLASSIC>();
        break;
      case ColorScheme::HOT_IRON:
        renderWithSampling<SamplesPerSide, ColorScheme::HOT_IRON>();
        break;
      case ColorScheme::ELECTRIC_BLUE:
        renderWithSampling<SamplesPerSide, ColorScheme::ELECTRIC_BLUE>();
        break;
      case ColorScheme::SUNSET:
        renderWithSampling<SamplesPerSide, ColorScheme::SUNSET>();
        break;
      case ColorScheme::GRAYSCALE:
        renderWithSampling<SamplesPerSide, ColorScheme::GRAYSCALE>();
        break;
      case ColorScheme::BLUE_WHITE:
        renderWithSampling<SamplesPerSide, ColorScheme::BLUE_WHITE>();
        break;
      case ColorScheme::EXPONENTIAL_LCH:
        renderWithSampling<SamplesPerSide, ColorScheme::EXPONENTIAL_LCH>();
        break;
      case ColorScheme::RAINBOW_SPIRAL:
        renderWithSampling<SamplesPerSide, ColorScheme::RAINBOW_SPIRAL>();
        break;
      case ColorScheme::OCEAN_DEPTHS:
        renderWithSampling<SamplesPerSide, ColorScheme::OCEAN_DEPTHS>();
        break;
      case ColorScheme::LAVA_FLOW:
        renderWithSampling<SamplesPerSide, ColorScheme::LAVA_FLOW>();
        break;
      case ColorScheme::CHERRY_BLOSSOM:
        renderWithSampling<SamplesPerSide, ColorScheme::CHERRY_BLOSSOM>();
        break;
      case ColorScheme::NEON_CYBERPUNK:
        renderWithSampling<SamplesPerSide, ColorScheme::NEON_CYBERPUNK>();
        break;
      case ColorScheme::AUTUMN_FOREST:
        renderWithSampling<SamplesPerSide, ColorScheme::AUTUMN_FOREST>();
        break;
      case ColorScheme::COUNT:
        renderWithSampling<SamplesPerSide, ColorScheme::COUNT>();
        break;
      }
    };
    switch (samples_per_side) {
    case 1:
      dispatch_1.operator()<1>();
      break;
    case 2:
      dispatch_1.operator()<2>();
      break;
    case 3:
      dispatch_1.operator()<3>();
      break;
    case 4:
      dispatch_1.operator()<4>();
      break;
    default:
      dispatch_1.operator()<1>();
      break; // Runtime fallback
    }
  }

  template <int SamplesPerSide, ColorScheme colour>
  void renderWithSampling() {
    constexpr auto const samples_per_pixel = SamplesPerSide * SamplesPerSide;

    // Pre-calculate coordinate transformation constants
    const double scale = 3.0 / (zoom * std::min(current_width, current_height));
    const batch_d offset_x_batch = batch_d(current_width / 2.0);
    const batch_d offset_y_batch = batch_d(current_height / 2.0);
    const batch_d center_x_batch = batch_d(center_x);
    const batch_d center_y_batch = batch_d(center_y);
    const batch_d scale_batch = batch_d(scale);

    constexpr auto BUF_N = samples_per_pixel + (batch_d::size - 1) / batch_d::size;

    auto coordinate_generator = [&](std::size_t px_start, std::size_t px_end) {
      auto const needed_pixels = px_end - px_start;
      auto const needed_samples = needed_pixels * samples_per_pixel;
      auto const total_batches = (needed_samples + batch_d::size - 1) / batch_d::size;

      auto const sample_start = px_start * samples_per_pixel;

      auto iteration_buffer = std::array<xsimd::batch<double>, BUF_N>{};
      auto mag_buffer = std::array<xsimd::batch<double>, BUF_N>{};
      auto current_buf = 0uz;
      auto write = 0uz;
      auto read = 0uz;

      auto sample_base = sample_start;
      for (std::size_t i = 0; i != total_batches; ++i) {
        // coords
        auto const sample_index = iota_batch(sample_base);
        auto const pixel_index = sample_index / samples_per_pixel;
        auto const sub_sample_index = sample_index % samples_per_pixel;

        auto const px = xsimd::batch_cast<double>(pixel_index % current_width);
        auto const py = xsimd::batch_cast<double>(pixel_index / current_width);

        auto const sx = xsimd::batch_cast<double>((sub_sample_index % SamplesPerSide) + 1);
        auto const sy = xsimd::batch_cast<double>((sub_sample_index / SamplesPerSide) + 1);

        // Pre-calculated constant for subpixel sampling
        const auto sub_distance = batch_d{1.0 / (SamplesPerSide + 1)};
        auto const sub_x = px + sub_distance * sx;
        auto const sub_y = py + sub_distance * sy;

        auto const real = center_x_batch + (sub_x - offset_x_batch) * scale_batch;
        auto const imag = center_y_batch - (sub_y - offset_y_batch) * scale_batch;

        // mandelbrot
        auto [iter, mag] = mandelbrot_simd<MAX_ITER>(real, imag);
        iteration_buffer[current_buf] = xsimd::batch_cast<double>(iter);
        mag_buffer[current_buf] = mag;
        write += batch_d::size;

        // collect colour
        while (write - read >= samples_per_pixel) {

          xsimd::batch<double> r_acc = xsimd::batch<double>{};
          xsimd::batch<double> g_acc = xsimd::batch<double>{};
          xsimd::batch<double> b_acc = xsimd::batch<double>{};
          xsimd::batch<size_t> count_mask = xsimd::batch<size_t>{};
          size_t batch_count = (samples_per_pixel + batch_d::size - 1) / batch_d::size;
          for (std::size_t i = 0; i != batch_count; ++i) {
            std::size_t logical_index = read + i * batch_d::size;
            std::size_t buf_idx = (logical_index / batch_d::size) % BUF_N;
            std::size_t start_offset = logical_index % batch_d::size;
            const auto &iter_batch = iteration_buffer[buf_idx];
            const auto &mag_batch = mag_buffer[buf_idx];

            // Simplified masking - only process valid lanes within this batch
            auto const remaining_samples = samples_per_pixel - i * batch_d::size;
            auto const iota = iota_batch<size_t>(0u);
            auto const lanes = std::min(batch_d::size, remaining_samples);
            auto const mask =
                (xsimd::batch{start_offset} <= iota) && (iota < xsimd::batch{start_offset + lanes});

            auto const mask_d = xsimd::batch_bool_cast<double>(mask);

            // Smooth coloring using both iterations and escape magnitude (if enabled)
            batch_d final_iter;
            if (smooth_coloring_enabled) {
              auto escaped_mask = mag_batch > batch_d(4.0);
              auto smooth_iter =
                  iter_batch - xsimd::log2(xsimd::log2(mag_batch)) + xsimd::log2(xsimd::log2(4.0));
              final_iter = select(escaped_mask, smooth_iter, iter_batch);
            } else {
              final_iter = iter_batch; // Use raw iteration count
            }

            // Calculate normalized t for most color schemes (expensive logarithm)
            auto t = xsimd::log(final_iter + 1.0) / xsimd::log(static_cast<double>(MAX_ITER + 1));

            xsimd::batch<double> r, g, b;
            switch (colour) {
            case ColorScheme::CLASSIC:
              std::tie(r, g, b) = getClassicColor_simd(t);
              break;
            case ColorScheme::HOT_IRON:
              std::tie(r, g, b) = getHotIronColor_simd(t);
              break;
            case ColorScheme::ELECTRIC_BLUE:
              std::tie(r, g, b) = getElectricBlueColor_simd(t);
              break;
            case ColorScheme::SUNSET:
              std::tie(r, g, b) = getSunsetColor_simd(t);
              break;
            case ColorScheme::GRAYSCALE:
              std::tie(r, g, b) = getGrayscaleColor_simd(t);
              break;
            case ColorScheme::EXPONENTIAL_LCH:
              std::tie(r, g, b) =
                  getExponentialLCH_simd(final_iter); // Uses smooth iterations directly
              break;
            case ColorScheme::BLUE_WHITE:
              std::tie(r, g, b) = getBlueWhiteColor_simd(t);
              break;
            case ColorScheme::RAINBOW_SPIRAL:
              std::tie(r, g, b) = getRainbowSpiralColor_simd(t);
              break;
            case ColorScheme::OCEAN_DEPTHS:
              std::tie(r, g, b) = getOceanDepthsColor_simd(t);
              break;
            case ColorScheme::LAVA_FLOW:
              std::tie(r, g, b) = getLavaFlowColor_simd(t);
              break;
            case ColorScheme::CHERRY_BLOSSOM:
              std::tie(r, g, b) = getCherryBlossomColor_simd(t);
              break;
            case ColorScheme::NEON_CYBERPUNK:
              std::tie(r, g, b) = getNeonCyberpunkColor_simd(t);
              break;
            case ColorScheme::AUTUMN_FOREST:
              std::tie(r, g, b) = getAutumnForestColor_simd(t);
              break;
            default:
              // Error fallback - render white to make it obvious
              r = batch_d(1.0);
              g = batch_d(1.0);
              b = batch_d(1.0);
              break;
            }

            // Convert to sRGB space before accumulation for proper gamma-correct averaging
            auto srgb_r = gammaCorrect_simd(r);
            auto srgb_g = gammaCorrect_simd(g);
            auto srgb_b = gammaCorrect_simd(b);
            
            // Accumulate colors only for valid samples (masked samples are automatically 0)
            r_acc += select(mask_d, srgb_r, batch_d(0.0));
            g_acc += select(mask_d, srgb_g, batch_d(0.0));
            b_acc += select(mask_d, srgb_b, batch_d(0.0));
            count_mask += select(mask, xsimd::batch<size_t>(1), xsimd::batch<size_t>(0));
          }
          double r_sum = xsimd::reduce_add(r_acc);
          double g_sum = xsimd::reduce_add(g_acc);
          double b_sum = xsimd::reduce_add(b_acc);
          std::size_t count = xsimd::reduce_add(count_mask);
          
          // Average in sRGB space (samples already converted during accumulation)
          double srgb_r = r_sum / count;
          double srgb_g = g_sum / count;
          double srgb_b = b_sum / count;
          
          sf::Color final_color{
              static_cast<sf::Uint8>(std::clamp(255.0 * srgb_r, 0.0, 255.0)),
              static_cast<sf::Uint8>(std::clamp(255.0 * srgb_g, 0.0, 255.0)),
              static_cast<sf::Uint8>(std::clamp(255.0 * srgb_b, 0.0, 255.0))
          };

          std::size_t pixel_index = read / samples_per_pixel;
          std::size_t actual_pixel_idx = px_start + pixel_index;
          if (actual_pixel_idx < current_width * current_height) {
            std::size_t px = actual_pixel_idx % current_width;
            std::size_t py = actual_pixel_idx / current_width;
            image.setPixel(px, py, final_color);
          }

          read += samples_per_pixel;
        }

        // update for next batch
        current_buf = (current_buf + 1) % BUF_N;
        sample_base += batch_d::size;
      }
    };

    auto scheduler = stdexec::schedule(thread_pool->get_scheduler());
    stdexec::sender auto sender = stdexec::bulk_chunked(
        scheduler,
        stdexec::par,
        current_width * current_height,
        coordinate_generator
    );

    stdexec::sync_wait(sender);
  }

  // ===== UI MANAGEMENT =====
  void showLoadingIndicator() {
    window.clear();
    window.draw(sprite);
    if (is_rendering)
      drawLoadingIndicator();
    else if (is_panning)
      drawPanningIndicator();
    window.display();
  }

  void draw() {
    window.clear();
    window.draw(sprite);
    if (is_rendering)
      drawLoadingIndicator();
    else if (is_panning)
      drawPanningIndicator();
    if (show_help)
      drawHelpOverlay();
    window.display();
  }

  void drawLoadingIndicator() {
    sf::RectangleShape overlay(sf::Vector2f(current_width, current_height));
    overlay.setFillColor(sf::Color(0, 0, 0, 128));
    window.draw(overlay);

    static float rotation = 0.0f;
    rotation += SPINNER_ROTATION_INCREMENT;
    if (rotation >= MAX_ROTATION_DEGREES)
      rotation -= MAX_ROTATION_DEGREES;

    sf::RectangleShape spinner(sf::Vector2f(40, 5));
    spinner.setFillColor(sf::Color::White);
    spinner.setOrigin(20, 2.5f);
    spinner.setPosition(current_width / 2.0f, current_height / 2.0f);
    spinner.setRotation(rotation);
    window.draw(spinner);

    sf::RectangleShape spinner2(sf::Vector2f(40, 5));
    spinner2.setFillColor(sf::Color::White);
    spinner2.setOrigin(20, 2.5f);
    spinner2.setPosition(current_width / 2.0f, current_height / 2.0f);
    spinner2.setRotation(rotation + 90.0f);
    window.draw(spinner2);
  }

  void drawPanningIndicator() {
    sf::CircleShape indicator(8);
    indicator.setFillColor(sf::Color(255, 255, 255, 180));
    indicator.setPosition(current_width - 30, 15);
    window.draw(indicator);

    for (int i = 0; i < 3; ++i) {
      sf::RectangleShape line(sf::Vector2f(12, 2));
      line.setFillColor(sf::Color(100, 100, 100, 180));
      line.setPosition(current_width - 35, 20 + i * 5);
      window.draw(line);
    }
  }

  void drawHelpOverlay() {
    // Semi-transparent dark background
    sf::RectangleShape overlay(sf::Vector2f(current_width, current_height));
    overlay.setFillColor(sf::Color(0, 0, 0, 180));
    window.draw(overlay);

    // Help panel background with responsive scaling
    const float base_panel_width = 650.0f;
    const float base_line_spacing = 26.0f;
    const float base_padding = 80.0f;
    const float min_margin = 40.0f; // Minimum margin from screen edges
    
    // Calculate desired panel size
    float desired_panel_height = static_cast<float>(help_texts.size() * base_line_spacing + base_padding);
    float max_available_height = current_height - 2 * min_margin;
    float max_available_width = current_width - 2 * min_margin;
    
    // Calculate scaling factor to fit screen
    float height_scale = (desired_panel_height > max_available_height) ? 
                        (max_available_height / desired_panel_height) : 1.0f;
    float width_scale = (base_panel_width > max_available_width) ? 
                       (max_available_width / base_panel_width) : 1.0f;
    float scale_factor = std::min(height_scale, width_scale);
    
    // Apply scaling
    float panel_width = base_panel_width * scale_factor;
    float panel_height = desired_panel_height * scale_factor;
    float line_spacing = base_line_spacing * scale_factor;
    
    float panel_x = (current_width - panel_width) / 2.0f;
    float panel_y = (current_height - panel_height) / 2.0f;

    sf::RectangleShape panel(sf::Vector2f(panel_width, panel_height));
    panel.setPosition(panel_x, panel_y);
    panel.setFillColor(sf::Color(30, 30, 40, 240));
    panel.setOutlineColor(sf::Color(100, 100, 120));
    panel.setOutlineThickness(2.0f);
    window.draw(panel);

    // Draw help texts with scaling
    for (std::size_t i = 0; i < help_texts.size(); ++i) {
      auto &text = help_texts[i];
      
      // Store original properties
      sf::Vector2f original_pos = text.getPosition();
      unsigned int original_size = text.getCharacterSize();
      
      // Apply scaling to text size
      unsigned int scaled_size = static_cast<unsigned int>(original_size * scale_factor);
      scaled_size = std::max(scaled_size, 8u); // Minimum readable size
      text.setCharacterSize(scaled_size);
      
      // Calculate scaled position relative to panel
      float scaled_y = panel_y + (i * line_spacing) + 20.0f; // 20.0f for top padding
      float scaled_x = panel_x + (30.0f * scale_factor); // Scale left margin too
      text.setPosition(scaled_x, scaled_y);
      
      window.draw(text);
      
      // Restore original properties
      text.setCharacterSize(original_size);
      text.setPosition(original_pos);
    }

    // If no font loaded, show a simple message
    if (help_texts.empty() || help_texts[0].getFont() == nullptr) {
      sf::Text fallback_text;
      fallback_text.setString("Font not loaded - Help unavailable");
      fallback_text.setCharacterSize(static_cast<unsigned int>(20 * scale_factor));
      fallback_text.setFillColor(sf::Color::Red);
      fallback_text.setPosition(panel_x + (50.0f * scale_factor), panel_y + (50.0f * scale_factor));
      window.draw(fallback_text);
    }

  }

  void updateWindowTitle(long long render_time_ms) {
    static constexpr std::size_t ESTIMATED_TITLE_LENGTH = 128;
    
    std::ostringstream title_stream;
    title_stream.str().reserve(ESTIMATED_TITLE_LENGTH);
    
    const auto scheme_name = getColorSchemeName(current_color_scheme);
    title_stream << "Mandelbrot Viewer [" << scheme_name << "]";
    
    if (anti_aliasing_enabled) {
      const auto aa_samples = static_cast<int>(aa_level) * static_cast<int>(aa_level);
      title_stream << " AA:" << aa_samples << "x";
    } else {
      title_stream << " AA:Off";
    }
    
    title_stream << (smooth_coloring_enabled ? " Smooth:On" : " Smooth:Off");
    title_stream << " - " << render_time_ms << "ms";
    
    if (!show_help) {
      title_stream << " (Press H for help)";
    }
    
    window.setTitle(title_stream.str());
  }

  [[nodiscard]] constexpr std::string_view getColorSchemeName(ColorScheme scheme) const noexcept {
    switch (scheme) {
    case ColorScheme::CLASSIC:
      return "Ultra Fractal Classic";
    case ColorScheme::HOT_IRON:
      return "Hot Iron";
    case ColorScheme::ELECTRIC_BLUE:
      return "Electric Blue";
    case ColorScheme::SUNSET:
      return "Sunset";
    case ColorScheme::GRAYSCALE:
      return "Grayscale";
    case ColorScheme::BLUE_WHITE:
      return "Blue to White";
    case ColorScheme::EXPONENTIAL_LCH:
      return "Exponential LCH";
    case ColorScheme::RAINBOW_SPIRAL:
      return "Rainbow Spiral";
    case ColorScheme::OCEAN_DEPTHS:
      return "Ocean Depths";
    case ColorScheme::LAVA_FLOW:
      return "Lava Flow";
    case ColorScheme::CHERRY_BLOSSOM:
      return "Cherry Blossom";
    case ColorScheme::NEON_CYBERPUNK:
      return "Neon Cyberpunk";
    case ColorScheme::AUTUMN_FOREST:
      return "Autumn Forest";
    default:
      return "Unknown";
    }
  }
};

int main() {
  MandelbrotViewer viewer;
  viewer.run();
  return 0;
}