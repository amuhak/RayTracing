#include "main.cuh"

__global__ void render(float *fb, int max_x, int max_y) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) {
        return;
    }
    const uint32_t pixel_index = j * max_x + i;
    const float    r           = static_cast<float>(i) / static_cast<float>(max_x - 1);
    const float    g           = static_cast<float>(j) / static_cast<float>(max_y - 1);
    const float    b           = 0.25f;

    fb[pixel_index * 3 + 0] = r;
    fb[pixel_index * 3 + 1] = g;
    fb[pixel_index * 3 + 2] = b;
}

int main() {
    // Image

    constexpr uint32_t image_width      = 1200;
    constexpr uint32_t image_height     = 600;
    constexpr uint32_t number_of_pixels = image_width * image_height;
    constexpr uint32_t buffer_size      = number_of_pixels * 3;
    float             *fb{nullptr};

    // Allocate FB
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&fb), buffer_size * sizeof(float)));

    // Render
    constexpr uint32_t x_block_size = 8;
    constexpr uint32_t y_block_size = 8;

    constexpr dim3 blocks(image_width / x_block_size + 1, image_height / y_block_size + 1);
    constexpr dim3 threads(x_block_size, y_block_size);

    render<<<blocks, threads>>>(fb, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Write to output.ppm

    std::ofstream output("output.ppm");

    output << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            auto idx = 3 * (j * image_width + i);
            auto r   = fb[idx++];
            auto g   = fb[idx++];
            auto b   = fb[idx];

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            output << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    output.close();
}
