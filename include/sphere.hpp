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
    sphere(const point3 &center, const double radius, std::shared_ptr<material> mat) :
        center(center), radius(std::fmax(0, radius)), mat(std::move(mat)) {}

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override;

private:
    point3                    center;
    double                    radius;
    std::shared_ptr<material> mat;
};


#endif // SPHERE_H
