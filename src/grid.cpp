//
// Created by amuhak on 4/7/2025.
//
#ifdef _MSC_VER
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include <stb_image_write.h>
#include "grid.hpp"

thread_local uint16_t grid::done{0};

void grid::write(std::ostream &out) const {
    std::string ans;
    for (size_t i = 0; i < data.size(); i += MAX_STRING_SIZE) {
        for (size_t j = i; j < std::min(i + MAX_STRING_SIZE, data.size()); j += 4) {
            write_color(ans, data[j], data[j + 1], data[j + 2]);
        }
        out << ans;
        ans.clear();
    }
    constexpr int channels = 4;
    const int stride = static_cast<int>(width) * channels;
    stbi_write_png("image.png", static_cast<int>(width), static_cast<int>(height), channels, data.data(), stride);
}

void grid::set(const uint32_t x, const uint32_t y, const color &c) {
    const size_t idx = (width * x + y) * 4; // 4 bytes per pixel (RGBA)
    data.at(idx + 3) = 255; // Set alpha to 255 (opaque)
    const auto &[r, g, b] = convert_color(c);
    // Dont need to range check since +3 is valid
    data[idx] = r; // Red
    data[idx + 1] = g; // Green
    data[idx + 2] = b; // Blue
    done++;
    if (done >= DONE_THRESHOLD) {
        total_done += done;
        done = 0;
    }
}