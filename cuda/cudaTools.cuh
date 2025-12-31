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

__global__ void render_init(const int64_t max_x, const int64_t max_y, curandState *rand_state) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) {
        return;
    }
    const int64_t pixel_index = j * max_x + i;
    curand_init(2025, pixel_index, 0, &rand_state[pixel_index]);
}


constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi       = std::numbers::pi_v<float>;

__host__ __device__ constexpr float degrees_to_radians(const float degrees) {
    return degrees * pi / 180.0f;
}


#endif // RAYTRACING_CUDATOOLS_CUH
