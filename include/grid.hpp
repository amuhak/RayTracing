//
// Created by amuhak on 4/7/2025.
//

#ifndef GRID_HPP
#define GRID_HPP
#include <atomic>
#include <iostream>
#include <vector>
#include "color.hpp"

constexpr size_t MAX_STRING_SIZE = 1000000;
constexpr int    DONE_THRESHOLD  = 10;
/**
 * This is a grid that holds the image data.
 */
class grid {
public:
    std::vector<uint8_t>         data;
    size_t                       width, height;
    size_t                       size;
    static thread_local uint16_t done;
    std::atomic_uint64_t         total_done{};

    grid(const size_t width, const size_t height) {
        data.resize(width * height * 4, 0); // 4 bytes per pixel (RGBA)
        this->width  = width;
        this->height = height;
        this->size   = width * height;
    }

    /**
     * @param x the row number
     * @param y the column number
     * @param c the color to set
     */
    void set(uint32_t x, uint32_t y, const color &c);

    void write(std::ostream &out) const;
};


#endif // GRID_HPP
