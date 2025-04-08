//
// Created by amuhak on 4/7/2025.
//

#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <limits>
#include <memory>

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

constexpr double degrees_to_radians(const double degrees) {
    return degrees * pi / 180.0;
}

#endif //RTWEEKEND_H
