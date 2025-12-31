#include "main.cuh"

float gamma_fix(const float input) {
    return std::sqrt(std::max(0.0f, input));
}

int main() {
    // Image
    constexpr float    viewport_height  = 2.0;
    constexpr float    focal_length     = 1.0;
    constexpr float    aspect_ratio     = 16.0f / 9.0f;
    constexpr uint32_t image_width      = 1920;
    constexpr uint32_t image_height     = std::max(1, static_cast<int>(image_width / aspect_ratio));
    constexpr uint32_t number_of_pixels = image_width * image_height;
    constexpr uint32_t buffer_size      = number_of_pixels * 4;
    constexpr float    viewport_width   = viewport_height * (static_cast<float>(image_width) / image_height);
    constexpr auto     camera_center    = point3(0, 0, 0);
    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    constexpr auto viewport_u = vec3(viewport_width, 0, 0);
    constexpr auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    constexpr auto pixel_delta_u = viewport_u / image_width;
    constexpr auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    constexpr auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    constexpr auto pixel00_loc         = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // CUDA configuration
    constexpr uint32_t x_block_size = 16;
    constexpr uint32_t y_block_size = 16;

    constexpr dim3 blocks(image_width / x_block_size + 1, image_height / y_block_size + 1);
    constexpr dim3 threads(x_block_size, y_block_size);

    float *fb{nullptr};

    // Allocate FB
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&fb), buffer_size * sizeof(float)));

    auto *fp4 = reinterpret_cast<float4 *>(fb);

    // Make the heap bigger for the world creation (128MB)
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 128));

    // Make the world
    hittable **d_world;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_world), sizeof(hittable *)));
    create_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialize the random number generator
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_rand_state), number_of_pixels * sizeof(curandState)));
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    const auto start_time = std::chrono::high_resolution_clock::now();

    render<<<blocks, threads>>>(fp4, image_width, image_height, pixel00_loc, pixel_delta_u, pixel_delta_v,
                                camera_center, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed  = end_time - start_time;
    std::cout << "Render time: " << std::chrono::duration<double, std::milli>(elapsed).count() << " ms\n";

    // Write to output.ppm

    std::ofstream output("output.ppm", std::ios::binary);

    output << "P6\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            auto idx = 4 * (j * image_width + i);
            auto r   = fb[idx++];
            auto g   = fb[idx++];
            auto b   = fb[idx];

            r = gamma_fix(r);
            g = gamma_fix(g);
            b = gamma_fix(b);

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            unsigned char pixel[3];
            pixel[0] = static_cast<unsigned char>(ir);
            pixel[1] = static_cast<unsigned char>(ig);
            pixel[2] = static_cast<unsigned char>(ib);

            output.write(reinterpret_cast<char *>(pixel), 3);
        }
    }

    output.close();
}
