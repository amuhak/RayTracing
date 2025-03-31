//
// Created by amuhak on 3/31/2025.
//

#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
    ray() = default;

    ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {
    }

    [[nodiscard]] const point3 &origin() const { return orig; }
    [[nodiscard]] const vec3 &direction() const { return dir; }

    [[nodiscard]] point3 at(const double t) const {
        return orig + t * dir;
    }

private:
    vec3 orig;
    vec3 dir;
};


#endif //RAY_H
