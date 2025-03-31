//
// Created by amuhak on 3/31/2025.
//

#ifndef VEC3_H
#define VEC3_H
#include <cmath>
#include <iostream>


class vec3 {
public:
    double e[3];

    vec3() : e{0, 0, 0} {
    }

    vec3(const double e0, const double e1, const double e2) : e{e0, e1, e2} {
    }

    [[nodiscard]] double x() const { return e[0]; }
    [[nodiscard]] double y() const { return e[1]; }
    [[nodiscard]] double z() const { return e[2]; }

    vec3 operator-() const { return {-e[0], -e[1], -e[2]}; }
    double operator[](const int i) const { return e[i]; }
    double &operator[](const int i) { return e[i]; }

    vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3 &operator*=(const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3 &operator/=(double t) {
        return *this *= 1 / t;
    }

    [[nodiscard]] double length() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] double length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions
#endif //VEC3_H
