//
// Created by amuhak on 4/7/2025.
//

#ifndef GRID_HPP
#define GRID_HPP
#include <atomic>
#include "color.hpp"

constexpr size_t MAX_STRING_SIZE = 1000000;
constexpr int DONE_THRESHOLD = 10;
/**
 * This is a grid that holds the image data.
 */
class grid {
public:
    std::vector<uint8_t> data;
    size_t width, height;
    size_t size;
    static thread_local uint16_t done;
    std::atomic_uint64_t total_done{};

    grid(const size_t width, const size_t height) {
        data.resize(width * height * 4, 0); // 4 bytes per pixel (RGBA)
        this->width = width;
        this->height = height;
        this->size = width * height;
    }

    /**
     * @param x the row number
     * @param y the column number
     * @param c the color to set
     */
    void set(const uint32_t x, const uint32_t y, const color &c) {
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

    void write(std::ostream &out) const {
        std::string ans;
        for (size_t i = 0; i < data.size(); i += MAX_STRING_SIZE) {
            for (size_t j = i; j < std::min(i + MAX_STRING_SIZE, data.size()); j += 4) {
                write_color(ans, data[j], data[j + 1], data[j + 2]);
            }
            out << ans;
            ans.clear();
        }
    }
};


#endif //GRID_HPP
