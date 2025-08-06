//
// Created by amuhak on 3/31/2025.
//

#include "vec3.hpp"

std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

vec3 operator+(const vec3 &u, const vec3 &v) {
    return {u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
}

vec3 operator-(const vec3 &u, const vec3 &v) {
    return {u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]};
}

vec3 operator*(const vec3 &u, const vec3 &v) {
    return {u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]};
}

vec3 operator*(const double t, const vec3 &v) {
    return {t * v.e[0], t * v.e[1], t * v.e[2]};
}

vec3 operator*(const vec3 &v, const type t) {
    return t * v;
}

vec3 operator/(const vec3 &v, const type t) {
    return 1 / t * v;
}

type dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
           + u.e[1] * v.e[1]
           + u.e[2] * v.e[2];
}

vec3 cross(const vec3 &u, const vec3 &v) {
    return {
        u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]
    };
}

vec3 unit_vector(const vec3 &v) {
    return v / v.length();
}

vec3 random_unit_vector() {
    while (true) {
        vec3 p = vec3::unit_random();
        if (const auto lensq = p.length_squared(); 1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

vec3 random_on_hemisphere(const vec3 &normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    return -on_unit_sphere;
}

vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(v, n) * n;
}