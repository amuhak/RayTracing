//
// Created by amuhak on 3/31/2025.
//

#ifndef VEC3_H
#define VEC3_H
using type = double;
#include <cmath>
#include <iostream>


class vec3 {
public:
    type e[3];

    vec3() : e{0, 0, 0} {
    }

    vec3(const type e0, const type e1, const type e2) : e{e0, e1, e2} {
    }

    [[nodiscard]] type x() const { return e[0]; }
    [[nodiscard]] type y() const { return e[1]; }
    [[nodiscard]] type z() const { return e[2]; }

    vec3 operator-() const { return {-e[0], -e[1], -e[2]}; }
    type operator[](const int i) const { return e[i]; }
    type &operator[](const int i) { return e[i]; }

    vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3 &operator*=(const type t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3 &operator/=(type t) {
        return *this *= 1 / t;
    }

    [[nodiscard]] type length() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] type length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions
#endif //VEC3_H
