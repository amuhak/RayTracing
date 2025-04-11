//
// Created by amuhak on 4/7/2025.
//
#include "camera.hpp"

#include "grid.hpp"

void camera::render(const hittable &world, const size_t threads = 10) {
    initialize();
    std::ofstream file;
    file.open("image.ppm");
    grid img(image_width, image_height);
    file << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    const size_t image_total{image_width * image_height};
    const size_t size = image_total / threads;
    for (size_t i = 0; i < image_total; i += size) {
        render_range(i, std::min(i + size, image_total), world, img);
    }
    // render_range(0, image_total, world, img);
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
    const int j{static_cast<int>(idx / image_width)};
    const int i{static_cast<int>(idx % image_width)};
    if (i == 0) {
        std::stringstream s;
        s << "\rWorking on scanline: " << std::setw(16) << j << "   " << std::flush;
        std::cout << s.str();
    }
    const auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
    const auto ray_direction = pixel_center - center;
    const ray r(center, ray_direction);
    const color pixel_color = ray_color(r, world);
    img.set(j, i, pixel_color);
}


void camera::initialize() {
    hittable_list world;

    world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));

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

[[nodiscard]] color camera::ray_color(const ray &r, const hittable &world) {
    hit_record rec;
    if (world.hit(r, interval(0, infinity), rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    const auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}
