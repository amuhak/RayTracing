//
// Created by amuhak on 4/7/2025.
//

#ifndef GRID_HPP
#define GRID_HPP
#include <vector>

#include "color.hpp"

constexpr size_t MAX_STRING_SIZE = 1000000;
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
        std::string ans;
        for (size_t i = 0; i < data.size(); i += MAX_STRING_SIZE) {
            for (size_t j = i; j < std::min(i + MAX_STRING_SIZE, data.size()); j++) {
                write_color(ans, data[j]);
            }
            out << ans;
            ans.clear();
        }
    }
};


#endif //GRID_HPP
