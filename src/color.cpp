//
// Created by amuhak on 3/31/2025.
//

#include "color.hpp"
#include "interval.hpp"
#include <SFML/Graphics.hpp>

void write_color(std::ostream &out, const color &pixel_color) {
    const auto r = pixel_color.x();
    const auto g = pixel_color.y();
    const auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    static const interval intensity(0.000, 0.999);
    int const rbyte = static_cast<int>(256 * intensity.clamp(r));
    int const gbyte = static_cast<int>(256 * intensity.clamp(g));
    int const bbyte = static_cast<int>(256 * intensity.clamp(b));

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
    const auto rbyte = static_cast<int>(255.999 * r);
    const auto gbyte = static_cast<int>(255.999 * g);
    const auto bbyte = static_cast<int>(255.999 * b);

    char buffer[16];
    const int n = std::snprintf(buffer, sizeof(buffer), "%d %d %d\n", rbyte, gbyte, bbyte);
    ans.append(buffer, static_cast<size_t>(n));
}

void write_color(std::string &ans, const uint8_t a, const uint8_t b, const uint8_t c) {
    char buffer[16];
    const int n = std::snprintf(buffer, sizeof(buffer), "%d %d %d\n", a, b, c);
    ans.append(buffer, static_cast<size_t>(n));
}

std::tuple<uint8_t, uint8_t, uint8_t> convert_color(
    const color &pixel_color
) {
    const auto a = static_cast<uint8_t>(255.999 * std::sqrt(std::max(0.0, pixel_color.x())));
    const auto b = static_cast<uint8_t>(255.999 * std::sqrt(std::max(0.0, pixel_color.y())));
    const auto c = static_cast<uint8_t>(255.999 * std::sqrt(std::max(0.0, pixel_color.z())));
    return std::make_tuple(a, b, c);
}
