//
// Created by amuhak on 12/29/2025.
//

#ifndef RAYTRACING_CUDATOOLS_CUH
#define RAYTRACING_CUDATOOLS_CUH
#include <iostream>
#include <numbers>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)


inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '"
                  << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi       = std::numbers::pi_v<float>;

__host__ __device__ constexpr float degrees_to_radians(const float degrees) {
    return degrees * pi / 180.0f;
}


#endif // RAYTRACING_CUDATOOLS_CUH
