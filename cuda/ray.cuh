//
// Created by amuhak on 12/30/2025.
//

#ifndef RAYTRACING_RAY_CUH
#define RAYTRACING_RAY_CUH
#include "vec3.cuh"

class ray {
public:
    __device__ ray() {
    }

    __device__ ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {
    }

    __device__ const point3 &origin() const {
        return orig;
    }
    __device__ const vec3 &direction() const {
        return dir;
    }

    __device__ point3 at(const double t) const {
        return orig + t * dir;
    }

private:
    point3 orig;
    vec3   dir;
};


#endif // RAYTRACING_RAY_CUH
