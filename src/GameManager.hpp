#pragma once
#include "Forces.hpp"
#include "GameData.hpp"
#include "SFML/System/Vector2.hpp"
#include <cmath>
#include <entt/entt.hpp>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <taskflow/taskflow.hpp>
#include <vector>

namespace game {
struct GameManager {
  std::unique_ptr<sf::RenderWindow> window;
  sf::Clock clock;
  entt::registry registry;
  tf::Executor executor;
  std::function<sf::Vector2f(sf::Vector2f)> force_function;
  float deltaTime = 1.0f / 60.0f;
  float accumulatorTime = 0.0f;
  float temperature = 3;
  float zeta = 0.0;
  float gamma = 10.0;
  float xi = 0.00;
  float Q = 5.0f;
  bool useThermostat = true;
  bool thermoNoise = false;
  Forces::RandomForce<5> rf;
  sf::Text statusText;
  sf::Font font;
  size_t timeWarp = 1;
  std::vector<std::mt19937> rnd;
  const size_t numParticles = 800;
  const size_t width = 800, height = 600;

  GameManager() {
    if (!font.loadFromFile("./resources/Inconsolata-Regular.ttf")) {
      throw std::runtime_error(
          "Couldn't load file ./resources/Inconsolata-Regular.ttf");
    }
    statusText.setFont(font);
    statusText.setFillColor(sf::Color::Red);
    statusText.setCharacterSize(24);
    sf::ContextSettings settings;
    settings.antialiasingLevel = 4;
    window = std::make_unique<sf::RenderWindow>(sf::VideoMode(width + 10, height + 10),
                                                "HelloSFML", sf::Style::Default,
                                                settings);
    clock = sf::Clock();
    rnd = std::vector<std::mt19937>(executor.num_workers());
    std::random_device rd;
    for (size_t i = 0; i < executor.num_workers(); i++) {
      rnd.push_back(std::mt19937(rd()));
    }
    registry = entt::registry();
    // force_function = [](sf::Vector2f dr) {
    //   const auto r = Forces::norm(dr);
    //   return 100.0f * Forces::normalize(dr) / (0.1f + r * r);
    // };
    rf = Forces::RandomForce<5>();
    // force_function = [this](sf::Vector2f dr) {return rf(dr); };
    force_function = [](sf::Vector2f dr) {
      return Forces::lj(dr, 0.0f, 10.0f, 2.5f);
    };
    // force_function = [](sf::Vector2f dr) {
    //   return Forces::vdw(dr, 5.0f, 5.0f, 0.2f);
    // };
    // force_function = [](sf::Vector2f dr) { return sf::Vector2f();};
    // force_function = [](sf::Vector2f dr) {
    //   const auto r = 5.0f * Forces::norm(dr);
    //   const auto e = Forces::normalize(dr);
    //   const auto t1 = -1.5f / powf(1.5f + r, 5.0f / 2);
    //   const auto t2 = 1.0f / powf(7.0f / 4 + r, 2.0f);
    //   return 15.0f * e * (t1 + t2);
    // };
    createWorld();
  }

  void createWorld() {

    std::uniform_real_distribution<float> dist_x(0, width);
    std::uniform_real_distribution<float> dist_y(0, height);

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

          if (Forces::norm(pos - posOther) < 15.0f) {
            pos = GameData::Position(dist_x(rnd[0]), dist_y(rnd[0]));
            checkCollision = true;
          }
        }
      }
    } while (checkCollision);
  }

  void pollEvents() {
    sf::Event event;
    while (window->pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window->close();
      }
    }
  }

  void draw(sf::CircleShape &circle, const GameData::Position &position) {
    circle.setPosition(position);
    window->draw(circle);
  }

  void render() {
    window->clear(sf::Color::Black);
    auto view = registry.view<sf::CircleShape, const GameData::Position>();

    for (auto [entity, circle, position] : view.each()) {
      draw(circle, position);
    }

    statusText.setString(fmt::format("Target temp = {:4f}", temperature));
    window->draw(statusText);
    window->display();
  }

  void updateForces() {
    auto viewForce = registry.view<const GameData::Particle,
                                   const GameData::Position, GameData::Force>();

    const auto viewOther =
        registry.view<const GameData::Particle, const GameData::Position>();
    tf::Taskflow taskflow;
    const auto ff = force_function;
    const float w = width;
    const float h = height;
    taskflow.for_each(viewForce.begin(), viewForce.end(),
                      [&viewForce, &viewOther, &ff, w, h](auto entity) {
                        auto [p1, x1, force] = viewForce.get(entity);
                        const auto r = x1 - sf::Vector2f(0.5f * w, 0.5f * h);

                        force = GameData::Force();
                        force += -0.02f * r;
                        for (auto [other, p2, x2] : viewOther.each()) {
                          if (other == entity) {
                            continue;
                          }
                          force += ff(x1 - x2);
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
    const auto dt = deltaTime;
    const auto w = width;
    const auto h = height;

    const float targetEnergy = 2 * numParticles * temperature;
    std::atomic<float> actualEnergy = 0.0;
    bool thermos = useThermostat;
    taskflow.for_each(
        view.begin(), view.end(),
        [&view, dt, w, h, &actualEnergy, thermos](auto entity) {
          auto [particle, position, momentum, force] = view.get(entity);
          if (thermos) {
            const auto deltaEnergy =
                (momentum.x * momentum.x + momentum.y + momentum.y) /
                particle.mass;
            auto energy = actualEnergy.load();
            while (!actualEnergy.compare_exchange_weak(energy,
                                                       energy + deltaEnergy))
              ;
          }
          position += dt * momentum / particle.mass;
          while (outOfBonds(position, (float)w, (float)h)) {
            reflect(position, momentum, (float)w, (float)h);
          }
        });

    executor.run(taskflow).wait();
    if (useThermostat) {
      auto dzeta = (actualEnergy - targetEnergy) * deltaTime / Q;
      dzeta -= gamma * zeta * deltaTime;
      std::normal_distribution<float> dist(0.0, deltaTime);
      dzeta += sqrtf(2 * temperature * gamma / Q) * dist(rnd[0]);
      zeta += dzeta;
    } else {
      zeta = 0.0f;
    }
  }

  void integrateMomentum() {
    auto view = registry.view<GameData::Momentum, const GameData::Force>();
    tf::Taskflow taskflow;
    const auto dt = deltaTime;
    const auto xi = zeta;
    auto rptr = &rnd;
    auto eptr = &executor;
    taskflow.for_each(
        view.begin(), view.end(), [&view, dt, xi, this](auto entity) {
          auto [momentum, force] = view.get(entity);
          std::normal_distribution<float> dist(0.0, xi * dt);
          const auto dx = thermoNoise ? dist(rnd[executor.this_worker_id()]) : 0.0f;
          const auto dy = thermoNoise ? dist(rnd[executor.this_worker_id()]) : 0.0f;
          momentum += dt * (force - xi * momentum) + sf::Vector2f(dx, dy);
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
      currentTime = std::min(currentTime, dst / spd);
    }
    auto dxi = abs((actualEnergy - targetEnergy)/ Q - gamma * zeta);
    if(dxi > 0.00001f && useThermostat) {
      currentTime =
          std::min(currentTime, 1.0f / dxi);
    }

    deltaTime = currentTime;
  }

  void handleKeyboard() {
    float tempDelta = 0.0f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
      tempDelta += 50.0f * deltaTime;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
      tempDelta -= 50.0f * deltaTime;
    }
    temperature = std::min(temperature + tempDelta, 10000.0f);
    temperature = std::max(temperature, 0.001f);
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
    accumulatorTime = std::max(accumulatorTime, 3.0f/60.0f);
    while (accumulatorTime >= deltaTime && clock.getElapsedTime().asSeconds() < 1.0f/60.0f) {
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