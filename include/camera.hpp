//
// Created by amuhak on 4/7/2025.
//

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "grid.hpp"
#include "color.hpp"
#include "hittable.hpp"
#include "main.hpp"
#include "camera.hpp"


constexpr int samples_per_pixel = 100;
constexpr int max_depth = 50; // Maximum number of ray bounces
constexpr double pixel_samples_scale{1.0 / samples_per_pixel}; // Color scale factor for a sum of pixel samples

class camera {
public:
    /* Public Camera Parameters Here */
    double aspect_ratio = 16.0 / 9.0; // Ratio of image width over height
    size_t image_width = 400; // Rendered image width in pixel count

    void render(const hittable &world, size_t threads = 1);

private:
    /* Private Camera Variables Here */
    size_t image_height{1}; // Rendered image height
    point3 center; // Camera center
    point3 pixel00_loc; // Location of pixel 0, 0
    vec3 pixel_delta_u; // Offset to pixel to the right
    vec3 pixel_delta_v; // Offset to pixel below

    void initialize();

    void render_range(size_t start, size_t end, const hittable &world, grid &img) const;

    void render_pixel(size_t idx, const hittable &world, grid &img) const;

    [[nodiscard]] ray get_ray(int i, int j) const;

    [[nodiscard]] vec3 sample_square() const;

    [[nodiscard]] static color ray_color(const ray &r, int depth, const hittable &world);
};


#endif //CAMERA_HPP
