//
// Created by amuhak on 4/7/2025.
//

#ifndef CAMERA_HPP
#define CAMERA_HPP
#include "color.hpp"
#include "hittable.hpp"
#include "main.h"
#include "camera.hpp"


class camera {
public:
    /* Public Camera Parameters Here */
    double aspect_ratio = 16.0 / 9.0; // Ratio of image width over height
    int image_width = 400; // Rendered image width in pixel count

    void render(const hittable &world);

private:
    /* Private Camera Variables Here */
    int image_height{1}; // Rendered image height
    point3 center; // Camera center
    point3 pixel00_loc; // Location of pixel 0, 0
    vec3 pixel_delta_u; // Offset to pixel to the right
    vec3 pixel_delta_v; // Offset to pixel below

    void initialize();

    [[nodiscard]] static color ray_color(const ray &r, const hittable &world);
};


#endif //CAMERA_HPP
