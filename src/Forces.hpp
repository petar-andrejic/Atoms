#pragma once
#include <random>
#include <SFML/System/Vector2.hpp>
#include <array>
#include <cmath>
#include <boost/math/special_functions/laguerre.hpp>


namespace Forces {
inline float norm(const sf::Vector2f &v) {
  return sqrtf(v.x * v.x + v.y * v.y);
}

inline float pow3(const float &x) { return x * x * x; }

inline float pow2(const float &x) { return x * x; }

inline sf::Vector2f normalize(const sf::Vector2f &v) {
  const auto v0 = norm(v);
  if (v0 == 0.0f) {
    return sf::Vector2f();
  } else {
    return v / v0;
  }
}

inline sf::Vector2f vdw(sf::Vector2f dr, float r1, float r2, float a) {
  const auto r = norm(dr);

  const auto num = 64 * pow3(r1) * pow3(r2);
  const auto den =
      pow2(pow2(r) - pow2(r1 + r2)) * pow2(pow2(r) - pow2(r1 - r2));
  return a * dr * num / (6 * den);
}

inline sf::Vector2f lj_cutoff(sf::Vector2f dr, float sigma, float a) {
  const auto x = (norm(dr) / sigma + 0.92293f);
  return 6 * a * normalize(dr) * (2 * powf(x, -13) - powf(x, -7));
}

inline sf::Vector2f lj(sf::Vector2f dr, float r0, float sigma, float a) {
  const auto x = (norm(dr) / sigma + r0);
  return 6 * a * normalize(dr) * (2 * powf(x, -13) - powf(x, -7));
};

template <size_t N> struct RandomForce {
  std::array<float, N> coeffs;
  float sigma = 5.0f;


  RandomForce() {
    std::random_device device;
    std::mt19937 rnd(device());
    std::normal_distribution<float> dist(0.0, 2.0);

    for (size_t i = 1; i < N; i++) {
      coeffs[i] = dist(rnd);
    }
  }

  sf::Vector2f operator()(sf::Vector2f dr) const {
    auto force = sf::Vector2f();
    const auto v = normalize(dr);
    const auto r = sigma * norm(dr);
    auto x = 1.0f;
    for (size_t i = 0; i < N; i++) {
      force += (coeffs[i] * x) * v;
      x *= -r;
    }
  
    return force * std::expf(-0.5f * r);
  }
  

};

inline sf::Vector2f hertzian_sphere(sf::Vector2f dr, float sigma, float eps) 
{
  const auto r = norm(dr);
  if(r >= sigma) return sf::Vector2f();
  return -5.0f * normalize(dr) * powf(1.0f - r/sigma, 1.5f) / (2.0f * sigma);
}

inline sf::Vector2f soft_coulomb(sf::Vector2f dr, float r0) 
{
  const auto r = norm(dr);
  return -normalize(dr) / pow2(r0 + r);
}

inline sf::Vector2f fake_harmonic(sf::Vector2f dr, float k, float r0) {
  const auto r = norm(dr);
  const auto x = r - r0;
  return -k * normalize(dr) * x * expf(-x);
}

inline sf::Vector2f harmonic(sf::Vector2f dr, float k, float r0) {
  const auto r = norm(dr);
  const auto x = r - r0;
  return -k * normalize(dr) * x;
}

inline sf::Vector2f screened_coulomb(sf::Vector2f dr, float r0, float alpha) 
{
  const auto r = norm(dr);
  return -normalize(dr) * expf(-alpha * r) *( powf(r0 + r, -2.0) + powf(r0 + r, -1.0));
}

} // namespace Forces