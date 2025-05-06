cmake_minimum_required(VERSION 3.28)
project(RayTracing)
option(GPU "Enables gpu support" "OFF")
option(TESTING "Switch to testing builds" "OFF")

# set(TESTING "ON")

message(STATUS "GPU is ${GPU}")
message(STATUS "Testing is ${TESTING}")

if (TESTING STREQUAL "ON")
    message(STATUS "Testing enabled")
    add_compile_definitions(TESTING)
endif ()

if (GPU STREQUAL "ON")
    find_program(CMAKE_CUDA_COMPILER nvcc)
    if (NOT CMAKE_CUDA_COMPILER)
        message(fatal_error "could not find nvcc. please set CMAKE_CUDA_COMPILER to the path of nvcc.
        for example: -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
    endif ()
    enable_language(CUDA)
    message(STATUS "CUDA enabled")
else ()
    message(STATUS "CUDA disabled")
endif ()

if (GPU STREQUAL "ON")
    project(RayTracing CUDA)
else ()
    project(RayTracing)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_MODE ON)
    message(STATUS "Debug mode")
else ()
    set(DEBUG_MODE OFF)
    message(STATUS "Release mode")
endif ()

if (DEBUG_MODE)
    if (GPU STREQUAL "ON")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 -Wall")
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES ".*[Ii][Nn][Tt][Ee][Ll].*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Wextra -Wconversion -Wsign-conversion -Woverloaded-virtual -pedantic -g -fno-inline -rdynamic")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Wextra -Wconversion -Wsign-conversion -Woverloaded-virtual -pedantic -g -fno-inline")
    endif ()
    add_compile_definitions(DEBUG)
else ()
    if (GPU STREQUAL "ON")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES ".*[Ii][Nn][Tt][Ee][Ll].*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -march=native  -flto -fwhole-program -fuse-linker-plugin -fprefetch-loop-arrays -fopenmp -fopenmp-simd")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -march=native")
    endif ()
endif ()

if (GPU STREQUAL "ON")
    set(CMAKE_CUDA_STANDARD 20)
else ()
    set(CMAKE_CXX_STANDARD 20)
endif ()

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
    )
endif ()

if (GPU STREQUAL "ON")
    set_target_properties(RayTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

if (GPU STREQUAL "ON")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif ()

target_include_directories(RayTracing PRIVATE ${PROJECT_SOURCE_DIR}/include)
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})