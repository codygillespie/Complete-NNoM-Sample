cmake_minimum_required(VERSION 3.21)
project(nnom_sample_c C)

set(CMAKE_C_STANDARD 23)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/library/nnom/src/backends/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/library/nnom/src/core/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/library/nnom/src/layers/*.c
)

add_executable(nnom_sample_c main.c
        ${SOURCES}
)
