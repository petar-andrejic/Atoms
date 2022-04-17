#include "GameManager.hpp"
#include <exception>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    auto app = game::GameManager();
    app.gameLoop();
  } catch(const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}