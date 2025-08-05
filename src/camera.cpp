//
// Created by amuhak on 4/7/2025.
//
#include "camera.hpp"
#include "grid.hpp"
#include "prettyPrint.h"
#include "display.hpp"
#include <thread>
#include <vector>
#include <atomic>

constexpr int WORK_PER_WORKER = 16; // number of pixels each worker will render in a single call
constexpr bool USE_PRETTY_PRINT = true; // whether to use pretty print or not
constexpr bool USE_DISPLAY = false; // whether to use display or not

void camera::render_worker(std::atomic<size_t> &next_pixel_idx, const hittable &world, grid &img) const {
    const size_t image_total = image_width * image_height;

    // Loop until all pixels are claimed
    while (true) {
        const size_t start_index = next_pixel_idx.fetch_add(WORK_PER_WORKER, std::memory_order_relaxed);

        if (start_index >= image_total) {
            break;
        }

        size_t end_index = std::min(start_index + WORK_PER_WORKER, image_total);

        for (size_t i = start_index; i < end_index; ++i) {
            render_pixel(i, world, img);
        }
    }
}

void camera::render(const hittable &world, const size_t threads) {
    initialize();
    std::ofstream file;
    file.open("image.ppm");
    grid img(image_width, image_height);
    file << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    std::atomic<size_t> next_pixel_idx = 0;

    std::vector<std::thread> thread_pool;
    thread_pool.reserve(threads);

    // Launch supporting threads
    std::optional<prettyPrint> printer;
    std::optional<std::thread> printer_thread;
    std::optional<display> disp;
    std::optional<std::thread> display_thread;

    if constexpr (USE_PRETTY_PRINT) {
        printer.emplace(img);
        printer_thread.emplace(&prettyPrint::run, &printer.value());
    }
    if constexpr (USE_DISPLAY) {
        disp.emplace();
        display_thread.emplace(&display::run,
                               &disp.value(),
                               static_cast<uint32_t>(image_width),
                               static_cast<uint32_t>(image_height),
                               std::ref(img.data)
        );
    }


    // Launch a fixed pool of worker threads
    for (size_t i = 0; i < threads; ++i) {
        thread_pool.emplace_back(&camera::render_worker, this, std::ref(next_pixel_idx), std::ref(world),
                                 std::ref(img));
    }

    // Join all worker threads
    for (auto &t: thread_pool) {
        t.join();
    }

    // Stop supporting threads
    if constexpr (USE_PRETTY_PRINT) {
        printer->keepUpdating = false;
        printer_thread->join();
    }
    if constexpr (USE_DISPLAY) {
        disp->keepUpdating = false;
        display_thread->join();
    }

    std::cout << "\nWriting image...     " << std::flush;
    img.write(file);
    std::cout << "\nDone.                 \n";
    file.close();
    std::cout << img.data.size() << " bytes written to image.ppm" << std::endl;
}

void camera::render_range(const size_t start, const size_t end, const hittable &world, grid &img) const {
    for (size_t i = start; i < end; i++) {
        render_pixel(i, world, img);
    }
}

void camera::render_pixel(const size_t idx, const hittable &world, grid &img) const {
    const uint32_t j{static_cast<uint32_t>(idx / image_width)};
    const uint32_t i{static_cast<uint32_t>(idx % image_width)};
    color pixel_color(0, 0, 0);
    for (int sample{}; sample < samples_per_pixel; sample++) {
        ray r = get_ray(i, j);
        pixel_color += ray_color(r, max_depth, world);
    }
    img.set(j, i, pixel_color * pixel_samples_scale);
}

void camera::initialize() {
    // Calculate the image height, and ensure that it's at least 1.
    image_height = std::max(static_cast<size_t>(1),
                            static_cast<size_t>(static_cast<double>(image_width) / aspect_ratio));

    // Viewport widths less than one are ok since they are real valued.
    constexpr double focal_length = 1.0;
    constexpr double viewport_height = 2.0;
    const double viewport_width = viewport_height * (
                                      static_cast<double>(image_width) / static_cast<double>(image_height)
                                  );
    center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    const auto viewport_u = vec3(viewport_width, 0, 0);
    const auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / static_cast<int>(image_width);
    pixel_delta_v = viewport_v / static_cast<int>(image_height);

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = center
                               - vec3(0, 0, focal_length)
                               - viewport_u / 2
                               - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
}

[[nodiscard]] color camera::ray_color(const ray &r, const int depth, const hittable &world) {
    if (depth <= 0) return {0, 0, 0};
    hit_record rec;
    if (world.hit(r, interval(0.001, infinity), rec)) {
        vec3 direction = random_on_hemisphere(rec.normal);
        // 0.5 is the reflection factor
        return 0.5 * ray_color(ray(rec.p, direction), depth - 1, world);
    }

    vec3 unit_direction = unit_vector(r.direction());
    const auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

[[nodiscard]] ray camera::get_ray(const uint32_t i, const uint32_t j) const {
    auto offset = sample_square();
    auto pixel_sample = pixel00_loc
                        + (i + offset.x()) * pixel_delta_u
                        + (j + offset.y()) * pixel_delta_v;

    auto ray_origin = center;
    auto ray_direction = pixel_sample - ray_origin;

    return {ray_origin, ray_direction};
}

[[nodiscard]] vec3 camera::sample_square() {
    return {random_double() - 0.5, random_double() - 0.5, 0};
}
