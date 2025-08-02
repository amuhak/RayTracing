//
// Created by amuhak on 4/7/2025.
//

#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <limits>
#include <memory>
#include <random>

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;
thread_local static std::mt19937_64 gen(std::random_device{}());
thread_local static std::uniform_real_distribution dis(0.0, 1.0);
thread_local static std::uniform_real_distribution dis1(-1.0, 1.0);


constexpr double degrees_to_radians(const double degrees) {
    return degrees * pi / 180.0;
}

inline double random_double() {
    return dis(gen);
}

inline double random_unit_double() {
    return dis1(gen);
}

inline double random_double(const double min, const double max) {
    std::uniform_real_distribution dis1(min, max);
    return dis1(gen);
}

#endif //RTWEEKEND_H
