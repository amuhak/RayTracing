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


class camera {
public:
    /* Public Camera Parameters Here */
    double aspect_ratio = 16.0 / 9.0; // Ratio of image width over height
    size_t image_width = 400; // Rendered image width in pixel count

    void render(const hittable &world, size_t threads);

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

    [[nodiscard]] static color ray_color(const ray &r, const hittable &world);
};


#endif //CAMERA_HPP
