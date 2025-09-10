//
// Created by amuhak on 4/5/2025.
//

#ifndef SPHERE_H
#define SPHERE_H

#include <cmath>
#include <memory>
#include <utility>
#include "hittable.hpp"

class sphere : public hittable {
public:
    // Stationary Sphere
    sphere(const point3 &static_center, double radius, std::shared_ptr<material> mat) :
        center(static_center, vec3(0, 0, 0)), radius(std::fmax(0, radius)), mat(mat) {
    }

    // Moving Sphere
    sphere(const point3 &center1, const point3 &center2, double radius, std::shared_ptr<material> mat) :
        center(center1, center2 - center1), radius(std::fmax(0, radius)), mat(mat) {
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override;

private:
    ray                       center;
    double                    radius;
    std::shared_ptr<material> mat;
};


#endif // SPHERE_H
