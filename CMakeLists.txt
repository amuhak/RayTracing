cmake_minimum_required(VERSION 3.28)

project(RayTracing LANGUAGES CXX)

option(GPU "Enables CUDA/GPU support" OFF)
option(TESTING "Enables testing-specific definitions" OFF)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(SFML COMPONENTS Graphics Window System CONFIG REQUIRED)
find_package(Stb REQUIRED)

if (GPU)
    # Enable CUDA
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if (NOT CMAKE_CUDA_COMPILER)
        message(fatal_error "could not find nvcc. please set CMAKE_CUDA_COMPILER to the path of nvcc.
        for example: -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
    endif ()
    message(STATUS "CUDA support: ENABLED")
else ()
    message(STATUS "CUDA support: DISABLED")
endif ()


# SOURCE FILES

if (GPU STREQUAL "ON")
    add_executable(RayTracing
            src/main.cu
    )
else ()
    add_executable(RayTracing
            src/main.cpp
            src/vec3.cpp
            src/color.cpp
            src/ray.cpp
            src/hittable.cpp
            src/sphere.cpp
            src/hittable_list.cpp
            src/interval.cpp
            src/camera.cpp
            src/grid.cpp
            src/prettyPrint.cpp
            src/rtweekend.cpp
            src/display.cpp
            src/material.cpp
    )
endif ()


# TARGETS

# target_include_directories(RayTracing PRIVATE ${Stb_INCLUDE_DIR})

# Add the header files
target_include_directories(RayTracing PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Link SFML libraries
target_link_libraries(RayTracing PRIVATE
        SFML::Graphics
        SFML::Window
        SFML::System
)

# Windows.h header fix
target_compile_definitions(RayTracing PRIVATE NOMINMAX)

if (TESTING)
    target_compile_definitions(RayTracing PRIVATE TESTING)
endif ()

# COMPILER FLAGS

set(WARNING_FLAGS
        -Wall -Wextra -Wconversion -Wsign-conversion
        -Woverloaded-virtual -pedantic
)

if (MSVC)
    # MSVC
    target_compile_options(RayTracing PRIVATE /W4 /permissive-
            $<$<CONFIG:Debug>:/Od /Zi>
            $<$<CONFIG:Release>:/O2 /GL /Ot /arch:AVX2>
    )
    target_link_options(RayTracing PRIVATE
            $<$<CONFIG:Release>:/LTCG>
    )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang
    target_compile_options(RayTracing PRIVATE ${COMMON_WARNING_FLAGS}
            $<$<CONFIG:Debug>:-g -O0>
            $<$<CONFIG:Release>:-O3 -march=native -flto=auto>
    )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC
    target_compile_options(RayTracing PRIVATE ${COMMON_WARNING_FLAGS}
            $<$<CONFIG:Debug>:-g -O0 -fno-inline>
            $<$<CONFIG:Release>:-O3 -march=native -flto=auto>
    )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel
    target_compile_options(RayTracing PRIVATE ${COMMON_WARNING_FLAGS}
            $<$<CONFIG:Debug>:-g -O0>
            $<$<CONFIG:Release>:-O3 -march=native>
    )
else ()
    message(WARNING "Unrecognized C++ compiler. Using generic flags.")
    target_compile_options(RayTracing PRIVATE
            $<$<CONFIG:Debug>:-g>
            $<$<CONFIG:Release>:-O3>
    )
endif ()

# CUDA Config
if (GPU)
    set_target_properties(RayTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(RayTracing PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G -g -O0 -Wall>>
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3>>
    )
endif ()