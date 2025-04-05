//
// Created by amuhak on 3/31/2025.
//

#include "color.h"

void write_color(std::ostream &out, const color &pixel_color) {
    const auto r = pixel_color.x();
    const auto g = pixel_color.y();
    const auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    const auto rbyte = static_cast<int>(255.999 * r);
    const auto gbyte = static_cast<int>(255.999 * g);
    const auto bbyte = static_cast<int>(255.999 * b);

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

void write_color(std::ostream &std_out, std::ostream &file, const color &pixel_color) {
    const auto r = pixel_color.x();
    const auto g = pixel_color.y();
    const auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    auto rbyte = static_cast<int>(255.999 * r);
    auto gbyte = static_cast<int>(255.999 * g);
    auto bbyte = static_cast<int>(255.999 * b);

    // Write out the pixel color components.
    std_out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
    file << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

void write_color(std::string &ans, const color &pixel_color) {
    const auto r = pixel_color.x();
    const auto g = pixel_color.y();
    const auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    auto rbyte = static_cast<int>(255.999 * r);
    auto gbyte = static_cast<int>(255.999 * g);
    auto bbyte = static_cast<int>(255.999 * b);

    char buffer[16];
    int n = std::snprintf(buffer, sizeof(buffer), "%d %d %d\n", rbyte, gbyte, bbyte);
    ans.append(buffer, n);
}
