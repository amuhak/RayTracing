//
// Created by amuhak on 12/30/2025.
//

#ifndef RAYTRACING_INTERVAL_CUH
#define RAYTRACING_INTERVAL_CUH

#include "cudaTools.cuh"

class interval {
public:
    float min, max;

    __device__ interval() : min(+infinity), max(-infinity) {
    }

    __device__ interval(float min, float max) : min(min), max(max) {
    }

    __device__ float size() const {
        return max - min;
    }

    __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    static const interval empty;
    static const interval universe;
};

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

#endif // RAYTRACING_INTERVAL_CUH
