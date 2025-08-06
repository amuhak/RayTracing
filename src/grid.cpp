//
// Created by amuhak on 4/7/2025.
//

#include "grid.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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
