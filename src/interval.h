//
// Created by amuhak on 4/7/2025.
//

#ifndef INTERVAL_H
#define INTERVAL_H
#include "rtweekend.h"


class interval {
public:
    double min, max;

    interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    interval(const double min, const double max) : min(min), max(max) {}

    [[nodiscard]] double size() const {
        return max - min;
    }

    [[nodiscard]] bool contains(const double x) const {
        return min <= x && x <= max;
    }

    [[nodiscard]] bool surrounds(const double x) const {
        return min < x && x < max;
    }

    static const interval empty;
    static const interval universe;
};

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);


#endif //INTERVAL_H
