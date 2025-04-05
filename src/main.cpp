#include <fstream>
#include <iomanip>
#include <iostream>
#include "color.h"
#include "ray.h"
#include "vec3.h"

double hit_sphere(const point3 &center, double radius, const ray &r) {
    const vec3 oc = center - r.origin();
    const auto a = r.direction().length_squared();
    const auto h = dot(r.direction(), oc);
    const auto c = oc.length_squared() - radius * radius;
    const auto discriminant = h * h - a * c;

    if (discriminant < 0) {
        return -1.0;
    }
    return (h - std::sqrt(discriminant)) / a;
}

color ray_color(const ray &r) {
    if (double t = hit_sphere(point3(0, 0, -1), 0.5, r); t > 0.0) {
        vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

int main() {
    // Image
    std::ofstream file;
    file.open("image.ppm");

    std::string ans = "";

    constexpr double aspect_ratio = 16.0 / 9.0;
    constexpr int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    constexpr int image_height = std::max(1, static_cast<int>(image_width / aspect_ratio));

    // Viewport widths less than one are ok since they are real valued.
    constexpr double focal_length = 1.0;
    constexpr double viewport_height = 2.0;
    constexpr double viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    const auto viewport_u = vec3(viewport_width, 0, 0);
    const auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    vec3 pixel_delta_u = viewport_u / image_width;
    vec3 pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                               - vec3(0, 0, focal_length)
                               - viewport_u / 2
                               - viewport_v / 2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Render
    ans.reserve(image_width * image_height * 16);
    ans += "P3\n"; // PPM header
    ans += std::to_string(image_width) + ' ' + std::to_string(image_height) + "\n"; // Image dimensions
    ans += "255\n"; // Max color value
    for (int j = 0; j < image_height; ++j) {
        std::cout << "\rScanlines remaining: " << std::setw(16) << image_height - j << "   " << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);
            color pixel_color = ray_color(r);
            write_color(ans, pixel_color);
        }
    }

    std::clog << "\rDone.                 \n";
    // std::cout << ans;
    file << ans;
    file.close();
    std::cout << ans.size() << " bytes written to image.ppm" << std::endl;
}
