//
// Created by amuhak on 12/29/2025.
//

#ifndef RAYTRACING_VEC3_CUH
#define RAYTRACING_VEC3_CUH

#include <cmath>
#include <iostream>
#include "cudaTools.cuh"

class vec3 {
public:
    float e[3];

    __host__ __device__ constexpr vec3() : e{0, 0, 0} {
    }
    __host__ __device__ constexpr vec3(const float e0, const float e1, const float e2) : e{e0, e1, e2} {
    }

    __host__ __device__ constexpr float x() const {
        return e[0];
    }
    __host__ __device__ constexpr float y() const {
        return e[1];
    }
    __host__ __device__ constexpr float z() const {
        return e[2];
    }

    __host__ __device__ constexpr vec3 operator-() const {
        return {-e[0], -e[1], -e[2]};
    }

    __host__ __device__ constexpr float operator[](int i) const {
        return e[i];
    }

    __host__ __device__ constexpr float &operator[](int i) {
        return e[i];
    }

    __host__ __device__ constexpr vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ constexpr vec3 &operator*=(const float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ constexpr vec3 &operator/=(const float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ float length() const {
        return std::sqrt(length_squared());
    }

    __host__ __device__ constexpr float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __device__ static float random_float(curandState &rand_state) {
        return curand_uniform(&rand_state);
    }

    __device__ static float random_float(const float min, const float max, curandState &rand_state) {
        return min + (max - min) * random_float(rand_state);
    }

    __device__ static vec3 random(curandState &rand_state) {
        return {random_float(rand_state), random_float(rand_state), random_float(rand_state)};
    }

    __device__ static vec3 random(const float min, const float max, curandState &rand_state) {
        return {random_float(min, max, rand_state), random_float(min, max, rand_state),
                random_float(min, max, rand_state)};
    }

    __device__ bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        constexpr float s{1e-8f};
        return std::fabs(e[0]) < s && std::fabs(e[1]) < s && std::fabs(e[2]) < s;
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ constexpr vec3 operator+(const vec3 &u, const vec3 &v) {
    return {u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
}

__host__ __device__ constexpr vec3 operator-(const vec3 &u, const vec3 &v) {
    return {u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]};
}

__host__ __device__ constexpr vec3 operator*(const vec3 &u, const vec3 &v) {
    return {u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]};
}

__host__ __device__ constexpr vec3 operator*(float t, const vec3 &v) {
    return {t * v.e[0], t * v.e[1], t * v.e[2]};
}

__host__ __device__ constexpr vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ constexpr vec3 operator/(const vec3 &v, float t) {
    return (1 / t) * v;
}

__host__ __device__ constexpr float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ constexpr vec3 cross(const vec3 &u, const vec3 &v) {
    return {u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2], u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

__host__ __device__ inline vec3 unit_vector(const vec3 &v) {
    return v / v.length();
}

__device__ inline vec3 random_unit_vector(curandState &rand_state) {
    // Height of the point on the unit sphere [-1, 1]
    float h = vec3::random_float(-1.0f, 1.0f, rand_state);

    // Angle around the circle [0, 2π]
    float a = vec3::random_float(0.0f, 2.0f * pi, rand_state);

    // Find the radius at height h, inside the unit sphere
    float r = sqrtf(1.0f - h * h);

    // Find the x, y coordinates from the angle and radius
    float x, y;
    __sincosf(a, &y, &x);
    return {r * x, r * y, h};
}

__device__ inline vec3 random_on_hemisphere(const vec3 &normal, curandState &rand_state) {
    const vec3 on_unit_sphere = random_unit_vector(rand_state);
    // If the random point is in the same hemisphere as the normal, return it
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    }
    // Else return the opposite point
    return -on_unit_sphere;
}

__device__ inline vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3 &uv, const vec3 &n, const float etai_over_etat) {
    const float cos_theta      = std::fminf(dot(-uv, n), 1.0f);
    vec3        r_out_perp     = etai_over_etat * (uv + cos_theta * n);
    vec3        r_out_parallel = -std::sqrtf(std::fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}


#endif // RAYTRACING_VEC3_CUH
