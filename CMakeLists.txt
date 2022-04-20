cmake_minimum_required(VERSION 3.0)

project(Atoms)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(SFML)
add_subdirectory(fmt)
add_subdirectory(entt)
add_subdirectory(taskflow)
add_subdirectory(delaunator-cpp)
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED)

file(GLOB SOURCES src/*.cpp)
add_executable(atoms ${SOURCES})

target_link_libraries(atoms PRIVATE sfml-graphics fmt::fmt-header-only EnTT::EnTT Taskflow Boost::boost delaunator)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources/
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resources/)