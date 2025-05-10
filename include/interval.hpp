//
// Created by amuhak on 4/7/2025.
//

#ifndef INTERVAL_H
#define INTERVAL_H
#include "rtweekend.hpp"


class interval {
public:
    double min, max;

    interval() : min(+infinity), max(-infinity) {
    } // Default interval is empty

    interval(const double min, const double max) : min(min), max(max) {
    }

    [[nodiscard]] double size() const {
        return max - min;
    }

    [[nodiscard]] bool contains(const double x) const {
        return min <= x && x <= max;
    }

    [[nodiscard]] bool surrounds(const double x) const {
        return min < x && x < max;
    }

    [[nodiscard]] double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const interval empty, universe;
};

#endif //INTERVAL_H
