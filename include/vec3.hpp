//
// Created by amuhak on 3/31/2025.
//

#ifndef VEC3_H
#define VEC3_H
using type = double;
#include "rtweekend.hpp"
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

    vec3 &operator/=(const type t) {
        return *this *= 1 / t;
    }

    [[nodiscard]] type length() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] type length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    static vec3 random() {
        return {random_double(), random_double(), random_double()};
    }

    static vec3 random(const double min, const double max) {
        return {random_double(min, max), random_double(min, max), random_double(min, max)};
    }

    static vec3 unit_random() {
        return {random_unit_double(), random_unit_double(), random_unit_double()};
    }

    bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        constexpr auto s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }
};

/**
 * Generate a random unit vector.
 * @return A random unit vector
 */
vec3 random_unit_vector();

/**
 * Generate a random vector that is in the same hemisphere as the normal.
 * @param normal The normal vector of the hemisphere
 * @return A random vector in the hemisphere defined by the normal
 */
vec3 random_on_hemisphere(const vec3 &normal);

vec3 reflect(const vec3 &v, const vec3 &n);

vec3 refract(const vec3 &uv, const vec3 &n, double etai_over_etat);

std::ostream &operator<<(std::ostream &out, const vec3 &v);

vec3 operator+(const vec3 &u, const vec3 &v);

vec3 operator-(const vec3 &u, const vec3 &v);

vec3 operator*(const vec3 &u, const vec3 &v);

vec3 operator*(double t, const vec3 &v);

vec3 operator*(const vec3 &v, type t);

vec3 operator/(const vec3 &v, type t);

/**
 * Calculate the dot product of two vectors.
 * @param u vector u
 * @param v vector v
 * @return The dot product of u and v
 */
type dot(const vec3 &u, const vec3 &v);

/**
 * Calculate the cross product of two vectors.
 * @param u vector u
 * @param v vector v
 * @return The cross product of u and v
 */
vec3 cross(const vec3 &u, const vec3 &v);

vec3 unit_vector(const vec3 &v);

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions
#endif //VEC3_H
