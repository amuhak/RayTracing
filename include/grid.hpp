//
// Created by amuhak on 4/7/2025.
//

#ifndef GRID_HPP
#define GRID_HPP
#include <vector>

#include "color.hpp"


/**
 * This is a grid that holds the image data.
 */
class grid {
public:
    std::vector<color> data;
    size_t width, height;
    size_t size;

    grid(const size_t width, const size_t height) {
        data.resize(width * height);
        this->width = width;
        this->height = height;
        this->size = width * height;
    }

    void set(const int idx, const color &c) {
        const auto i(static_cast<size_t>(idx));
        data.at(i) = c;
    }

    /**
     * @param x the row number
     * @param y the column number
     * @param c the color to set
     */
    void set(const int x, const int y, const color &c) {
        set(x * static_cast<int>(width) + y, c);
    }

    void write(std::ostream &out) const {
        for (const auto &i: data) write_color(out, i);
    }
};


#endif //GRID_HPP
