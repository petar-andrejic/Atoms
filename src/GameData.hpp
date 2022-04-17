#include "SFML/Graphics/Color.hpp"
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics.hpp>

namespace GameData {

struct Position : sf::Vector2f {
  using Vector2::Vector2;
};

struct Momentum : sf::Vector2f {
  using Vector2::Vector2;
};

struct Force : sf::Vector2f {
  using Vector2::Vector2;
};

struct Particle {
  float mass = 1.0f;
  float radius = 5.0f;
  sf::Color color = sf::Color::White;
};

} // namespace GameData