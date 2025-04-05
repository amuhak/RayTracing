//
// Created by amuhak on 3/31/2025.
//

#include "vec3.h"

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
    return (1 / t) * v;
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
