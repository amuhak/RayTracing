//
// Created by amuhak on 3/31/2025.
//

#ifndef COLOR_H
#define COLOR_H
#include "vec3.hpp"
#include <string>

using color = vec3;

void write_color(std::ostream &out, const color &pixel_color);

void write_color(std::ostream &std_out, std::ostream &file, const color &pixel_color);

void write_color(std::string &ans, const color &pixel_color);

void write_color(std::string &ans, uint8_t a, uint8_t b, uint8_t c);

std::tuple<uint8_t, uint8_t, uint8_t> convert_color(const color &pixel_color);

#endif //COLOR_H