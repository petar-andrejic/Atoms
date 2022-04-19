#pragma once
#include "Forces.hpp"
#include "GameData.hpp"
#include "SFML/Graphics/Color.hpp"
#include "SFML/System/Vector2.hpp"
#include "SFML/Window/Event.hpp"
#include "SFML/Window/Keyboard.hpp"
#include "entt/entity/fwd.hpp"
#include <cmath>
#include <entt/entt.hpp>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <taskflow/taskflow.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace game {
struct GridPos {
  int x = 0;
  int y = 0;
  GridPos(){};
  GridPos(int x, int y) : x(x), y(y){};
};

struct GameManager {
  std::unique_ptr<sf::RenderWindow> window;
  sf::Clock clock;
  entt::registry registry;
  tf::Executor executor;
  std::function<sf::Vector2f(sf::Vector2f)> force_function;
  int fontSize = 12;

  float deltaTime = 1.0f / 60.0f;
  float accumulatorTime = 0.0f;
  float temperature;
  float Q;
  float displayTemp;
  float zeta = 0.0;
  float gamma = 10.0;
  float xi = 0.00;

  bool useThermostat = true;
  bool useTrap = true;
  bool useThermoNoise = true;
  bool useGravity = false;

  Forces::RandomForce<5> rf;
  sf::Text thermostatStatusText;
  sf::Text thermostatTempText;
  sf::Text deltatimeStatusText;
  sf::Text trapStatusText;
  sf::Text gravityStatusText;

  GameData::Momentum averageMomentum;

  sf::Font font;
  size_t timeWarp = 1;
  std::vector<std::mt19937> rnd;
  size_t numParticles = 400;
  size_t cellSize = 200;
  size_t cellsX = 4, cellsY = 4;
  std::vector<std::vector<entt::entity>> grid; // using stride indexing
  float width, height;

  GameManager() {
    if (!font.loadFromFile("./resources/Inconsolata-Regular.ttf")) {
      throw std::runtime_error(
          "Couldn't load file ./resources/Inconsolata-Regular.ttf");
    }

    setupText();
    width = cellsX * cellSize;
    height = cellsY * cellSize;
    grid = std::vector<std::vector<entt::entity>>(cellsX * cellsY);

    sf::ContextSettings settings;
    settings.antialiasingLevel = 4;
    window = std::make_unique<sf::RenderWindow>(
        sf::VideoMode(cellsX * cellSize + 10, cellsY * cellSize + 10),
        "HelloSFML", sf::Style::Default, settings);
    clock = sf::Clock();
    rnd = std::vector<std::mt19937>(executor.num_workers());
    std::random_device rd;
    for (size_t i = 0; i < executor.num_workers(); i++) {
      rnd.push_back(std::mt19937(rd()));
    }
    registry = entt::registry();
    rf = Forces::RandomForce<5>();
    // rf.sigma = 5.0f;
    // force_function = [](sf::Vector2f dr) {
    //   return Forces::lj(dr, 0.0f, 10.0f, 2.5f);
    // };
    force_function = [](sf::Vector2f dr) {
      // return Forces::vdw(dr, 5.0f, 5.0f, 0.00004);
      auto force = -Forces::soft_coulomb(dr, 0.0f);
      force += Forces::lj(dr, 0.0f, 10.0f, 2.5f);
      // force += Forces::hertzian_sphere(dr, 10.0f, 12.0f);
      return force;
    };
    // force_function = [this](sf::Vector2f dr) {
    //   return rf(dr);
    // };
    setTemp(3.0f);
    createWorld();
  };

  const GridPos getGridLoc(const GameData::Position &position) const {
    int x = std::floorf(position.x / cellSize);
    int y = std::floorf(position.y / cellSize);
    if (x < 0)
      x = 0;
    if (y < 0)
      y = 0;
    if (x >= cellsX)
      x = cellsX - 1;
    if (y >= cellsY)
      y = cellsY - 1;

    return GridPos(x, y);
  };

  void setupText() {
    thermostatTempText.setFont(font);
    thermostatTempText.setFillColor(sf::Color::Red);
    thermostatTempText.setCharacterSize(fontSize);
    thermostatStatusText.setFont(font);
    thermostatStatusText.setFillColor(sf::Color::Red);
    thermostatStatusText.setCharacterSize(fontSize);
    deltatimeStatusText.setFont(font);
    deltatimeStatusText.setFillColor(sf::Color::Red);
    deltatimeStatusText.setCharacterSize(fontSize);
    trapStatusText.setFont(font);
    trapStatusText.setFillColor(sf::Color::Red);
    trapStatusText.setCharacterSize(fontSize);
    gravityStatusText.setFont(font);
    gravityStatusText.setFillColor(sf::Color::Red);
    gravityStatusText.setCharacterSize(fontSize);

    thermostatTempText.setPosition(0, fontSize);
    deltatimeStatusText.setPosition(0, 2 * fontSize);
    trapStatusText.setPosition(0, 3 * fontSize);
    gravityStatusText.setPosition(0, 4 * fontSize);
  };

  void updateGrid() {
    for (auto &list : grid) {
      list.clear();
    }

    auto view = registry.view<const GameData::Position, GridPos>();
    for (auto [entity, position, gridPos] : view.each()) {
      gridPos = getGridLoc(position);
      grid[gridPos.x + cellsX * gridPos.y].push_back(entity);
    }
  };

  std::vector<entt::entity> neighbours(entt::entity entity) const {
    const auto pos = registry.get<GridPos>(entity);
    std::vector<entt::entity> v;
    std::vector<GridPos> toCheck;
    toCheck.reserve(9);

    for (int i = -1; i < 2; i++) {
      for (int j = -1; j < 2; j++) {
        const auto newPos = GridPos(pos.x + i, pos.y + j);
        if (newPos.x < cellsX && newPos.x >= 0 && newPos.y < cellsY &&
            newPos.y >= 0) {
          toCheck.push_back(newPos);
        }
      }
    }

    for (const auto &newPos : toCheck) {
      for (auto e : grid[newPos.x + cellsX * newPos.y]) {
        v.push_back(e);
      }
    }

    return v;
  };

  void createWorld() {
    zeta = 0.0f;
    std::uniform_real_distribution<float> dist_x(0, width);
    std::uniform_real_distribution<float> dist_y(0, height);
    registry = entt::registry();
    for (size_t i = 0; i < numParticles; i++) {

      const auto entity = registry.create();
      const auto particle = registry.emplace<GameData::Particle>(entity);
      const auto position = registry.emplace<GameData::Position>(
          entity, dist_x(rnd[0]), dist_y(rnd[0]));
      std::normal_distribution<float> dist_p(
          0, std::sqrtf(particle.mass * temperature));
      const auto momentum = registry.emplace<GameData::Momentum>(
          entity, dist_p(rnd[0]), dist_p(rnd[0]));
      const auto force = registry.emplace<GameData::Force>(entity);
      registry.emplace<GridPos>(entity);
      auto circle = registry.emplace<sf::CircleShape>(entity, particle.radius);
      circle.setFillColor(particle.color);
    }

    auto view = registry.view<GameData::Position>();
    const auto viewOther = registry.view<const GameData::Position>();
    auto checkCollision = false;
    do {
      checkCollision = false;
      for (auto [entity, pos] : view.each()) {
        for (auto [other, posOther] : viewOther.each()) {
          if (entity == other) {
            continue;
          }

          if (Forces::norm(pos - posOther) < 10.0f) {
            pos = GameData::Position(dist_x(rnd[0]), dist_y(rnd[0]));
            checkCollision = true;
          }
        }
      }
    } while (checkCollision);

    updateGrid();
  }

  void pollEvents() {
    sf::Event event;
    while (window->pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window->close();
      }

      if (event.type == sf::Event::KeyPressed) {
        if (event.key.code == sf::Keyboard::T) {
          useThermostat = !useThermostat;
          zeta = 0.0f;
        }

        if (event.key.code == sf::Keyboard::F) {
          useTrap = !useTrap;
        }

        if (event.key.code == sf::Keyboard::G) {
          useGravity = !useGravity;
        }

        if (event.key.code == sf::Keyboard::R) {
          createWorld();
        }

        if (event.key.code == sf::Keyboard::N) {
          useThermoNoise = !useThermoNoise;
        }
      }
    }
  }

  void drawShape(sf::CircleShape &circle, const GameData::Position &position,
                 const GameData::Momentum &momentum) {
    const int r = std::floorf(
        255 * (1 - expf(-0.1f * Forces::norm(momentum - averageMomentum))));
    const int g =
        std::floorf(255 * (1 - expf(-0.25f * Forces::norm(averageMomentum))));
    circle.setFillColor(sf::Color(r, g, 255, 255));
    circle.setPosition(position);
    window->draw(circle);
  }

  void render() {
    window->clear(sf::Color::Black);
    auto view = registry.view<sf::CircleShape, const GameData::Position,
                              const GameData::Momentum>();
    averageMomentum = GameData::Momentum();
    displayTemp = 0.0f;
    for (auto [entity, circle, position, momentum] : view.each()) {
      averageMomentum += momentum;
      displayTemp += (momentum.x * momentum.x + momentum.y * momentum.y) /
                     (2 * numParticles);
    }
    averageMomentum /= (float)numParticles;

    for (auto [entity, circle, position, momentum] : view.each()) {
      drawShape(circle, position, momentum);
    }

    const auto thermoBool = useThermostat ? "ON" : "OFF";
    const auto noiseBool = useThermoNoise ? "ON" : "OFF";

    thermostatStatusText.setString(
        fmt::format("Thermostat: {}, Noise: {}", thermoBool, noiseBool));

    if (useTrap) {
      trapStatusText.setString("Harmonic trap: ON");
    } else {
      trapStatusText.setString("Harmonic trap: OFF");
    }

    if (useGravity) {
      gravityStatusText.setString("Gravity: ON");
    } else {
      gravityStatusText.setString("Gravity: OFF");
    }

    thermostatTempText.setString(fmt::format(
        "Target temp = {:4f}, Actual = {:4f}", temperature, displayTemp));
    deltatimeStatusText.setString(fmt::format("dt = {:4f} s", deltaTime));

    window->draw(thermostatStatusText);
    window->draw(thermostatTempText);
    window->draw(deltatimeStatusText);
    window->draw(trapStatusText);
    window->draw(gravityStatusText);
    window->display();
  }

  void setTemp(const float &newTemp) {
    temperature = newTemp;
    if (newTemp < 10.0f) {
      Q = 10.0f;
    } else {
      Q = newTemp;
    }
  }

  void updateForces() {
    auto viewForce = registry.view<const GameData::Particle,
                                   const GameData::Position, GameData::Force>();

    const auto viewOther =
        registry.view<const GameData::Particle, const GameData::Position>();
    tf::Taskflow taskflow;
    taskflow.for_each(viewForce.begin(), viewForce.end(),
                      [&viewForce, &viewOther, this](auto entity) {
                        auto [p1, x1, force] = viewForce.get(entity);
                        const auto r =
                            x1 - sf::Vector2f(0.5f * width, 0.5f * height);

                        force = GameData::Force();
                        if (useTrap) {
                          force += -0.02f * r;
                        }

                        if (useGravity) {
                          force += sf::Vector2f(0, 10);
                        }

                        const auto others = neighbours(entity);
                        for (auto other : others) {
                          if (other == entity) {
                            continue;
                          }
                          const auto x2 =
                              registry.get<GameData::Position>(other);
                          force += force_function(x1 - x2);
                        }
                      });
    executor.run(taskflow).wait();
  }

  static bool outOfBonds(sf::Vector2f &position, float width, float height) {
    return (position.x < 0) || (position.x > width) || (position.y < 0) ||
           (position.y > height);
  }

  static void reflect(sf::Vector2f &position, sf::Vector2f &velocity,
                      float width, float height) {
    if (position.x < 0) {
      position.x = -position.x;
      velocity.x = -velocity.x;
    } else if (position.x > width) {
      position.x = 2 * width - position.x;
      velocity.x = -velocity.x;
    }

    if (position.y < 0) {
      position.y = -position.y;
      velocity.y = -velocity.y;
    } else if (position.y > height) {
      position.y = 2 * height - position.y;
      velocity.y = -velocity.y;
    }
  }

  void integratePosition() {
    auto view = registry.view<const GameData::Particle, GameData::Position,
                              GameData::Momentum, const GameData::Force>();

    tf::Taskflow taskflow;

    const float targetEnergy = 2 * numParticles * temperature;
    std::atomic<float> actualEnergy = 0.0;
    taskflow.for_each(
        view.begin(), view.end(), [&view, &actualEnergy, this](auto entity) {
          auto [particle, position, momentum, force] = view.get(entity);
          if (useThermostat) {
            const auto deltaEnergy =
                (momentum.x * momentum.x + momentum.y * momentum.y) /
                particle.mass;
            auto energy = actualEnergy.load();
            while (!actualEnergy.compare_exchange_weak(energy,
                                                       energy + deltaEnergy))
              ;
          }
          position += deltaTime * momentum / particle.mass;
          while (outOfBonds(position, (float)width, (float)height)) {
            reflect(position, momentum, (float)width, (float)height);
          }
        });

    executor.run(taskflow).wait();
    updateGrid();
    if (useThermostat) {
      auto dzeta = (actualEnergy - targetEnergy) * deltaTime / Q;
      dzeta -= gamma * zeta * deltaTime;
      std::normal_distribution<float> dist(0.0, deltaTime);
      dzeta += sqrtf(2 * temperature * gamma / Q) * dist(rnd[0]);
      zeta += dzeta;
    }
  }

  void integrateMomentum() {
    auto view = registry.view<GameData::Momentum, const GameData::Force>();
    tf::Taskflow taskflow;
    taskflow.for_each(view.begin(), view.end(), [&view, this](auto entity) {
      auto [momentum, force] = view.get(entity);
      std::normal_distribution<float> dist(0.0, zeta * deltaTime);
      const auto p = Forces::norm(momentum);
      const auto dx =
          useThermoNoise ? dist(rnd[executor.this_worker_id()]) : 0.0f;
      const auto dy =
          useThermoNoise ? dist(rnd[executor.this_worker_id()]) : 0.0f;
      momentum += deltaTime * (force - zeta * momentum) + sf::Vector2f(dx, dy);
    });
    executor.run(taskflow).wait();
  }

  void updateDeltaTime() {
    auto currentTime = 1 / 60.0f;
    auto view =
        registry.view<const GameData::Particle, const GameData::Momentum>();

    const float targetEnergy = 2 * numParticles * temperature;
    float actualEnergy = 0.0;

    for (auto [entity, particle, momentum] : view.each()) {
      const float p2 = momentum.x * momentum.x + momentum.y + momentum.y;
      if (useThermostat) {
        actualEnergy += p2 / particle.mass;
      }
      const auto spd = sqrtf(p2) / particle.mass;
      const auto dst = 0.1f;
      if (spd > 0.00001f) {
        currentTime = std::min(currentTime, dst / spd);
      }
    }
    if (zeta > 0.00001f && useThermostat) {
      currentTime = std::min(currentTime, 1 / zeta);
    }
    auto dzeta = abs((actualEnergy - targetEnergy) / Q - gamma * zeta);
    if (dzeta > 0.00001f && useThermostat) {
      currentTime = std::min(currentTime, 10.0f / dzeta);
    }

    deltaTime = currentTime;
  }

  void handleKeyboard() {
    float tempDelta = 0.0f;
    float tempIncrement = 0.5f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift) ||
        sf::Keyboard::isKeyPressed(sf::Keyboard::RShift)) {
      tempIncrement = 100.0f;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
      tempDelta += tempIncrement * deltaTime;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
      tempDelta -= tempIncrement * deltaTime;
    }
    setTemp(std::min(temperature + tempDelta, 10000.0f));
    setTemp(std::max(temperature, 0.001f));
  }

  void fixedUpdate() {
    integrateMomentum();
    updateDeltaTime();
    integratePosition();
    updateForces();
    handleKeyboard();
  }

  void update() {
    const auto elapsed = clock.restart();
    accumulatorTime += elapsed.asSeconds();
    accumulatorTime = std::max(accumulatorTime, 1.0f / 60.0f);
    while (accumulatorTime >= deltaTime &&
           clock.getElapsedTime().asSeconds() < 1.0f / 60.0f) {
      fixedUpdate();
      accumulatorTime -= deltaTime;
    }
  }

  void gameLoop() {
    while (window->isOpen()) {
      pollEvents();
      update();
      render();
    }
  }
};

} // namespace game