//
// Created by amuhak on 4/7/2025.
//

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <atomic>
#include "color.hpp"
#include "vec3.hpp"


class hittable;
class ray;
class grid;

class camera {
public:
    // Public Camera Variables
    double aspect_ratio        = 16.0 / 9.0;              // Ratio of image width over height
    size_t image_width         = 400;                     // Rendered image width in pixel count
    int    samples_per_pixel   = 10;                      // Count of random samples for each pixel
    int    max_depth           = 50;                      // Maximum number of ray bounces into scene
    double pixel_samples_scale = 1.0 / samples_per_pixel; // Color scale factor for a sum of pixel samples
    double vfov                = 90;                      // Vertical view angle (field of view)
    point3 lookfrom            = point3(0, 0, 0);         // Point camera is looking from
    point3 lookat              = point3(0, 0, -1);        // Point camera is looking at
    vec3   vup                 = vec3(0, 1, 0);           // Camera-relative "up" direction
    /**
     * Constructor for the camera class.
     * @param world The hittable world to render
     * @param threads The number of threads to use for rendering
     */
    void render(const hittable &world, size_t threads = 1);

private:
    /* Private Camera Variables Here */
    size_t image_height{1}; // Rendered image height
    point3 center;          // Camera center
    point3 pixel00_loc;     // Location of pixel 0, 0
    vec3   pixel_delta_u;   // Offset to pixel to the right
    vec3   pixel_delta_v;   // Offset to pixel below
    vec3   u, v, w;         // Camera frame basis vectors

    /**
     * Initializes a variety of constants and variables used to render the images.
     */
    void initialize();

    void render_worker(std::atomic<size_t> &next_pixel_idx, const hittable &world, grid &img) const;

    /**
     * Renders a single pixel in the image.
     * @param idx the index of the pixel to render
     * @param world The hittable world to render
     * @param img The location to store the rendered image
     */
    void render_pixel(size_t idx, const hittable &world, grid &img) const;

    /**
     * Generates rays from the camera to random points around the pixel at (i, j).
     * @param i The i coordinate of the pixel.
     * @param j The j coordinate of the pixel.
     * @return A ray from the camera to a point.
     */
    [[nodiscard]] ray get_ray(uint32_t i, uint32_t j) const;

    /**
     * @return random x and y between -0.5 and 0.5 and a z of 0.
     */
    [[nodiscard]] static vec3 sample_square();

    /**
     * Recursively samples the color of a ray.
     * @param r The ray to trace.
     * @param depth The current depth of the ray.
     * @param world The hittable world to trace the ray against.
     * @return The color of the ray sampled.
     */
    [[nodiscard]] static color ray_color(const ray &r, int depth, const hittable &world);
};


#endif // CAMERA_HPP
