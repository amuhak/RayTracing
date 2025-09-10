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
        auto rvec = vec3(radius, radius, radius);
        bbox      = aabb(static_center - rvec, static_center + rvec);
    }

    // Moving Sphere
    sphere(const point3 &center1, const point3 &center2, double radius, std::shared_ptr<material> mat) :
        center(center1, center2 - center1), radius(std::fmax(0, radius)), mat(mat) {
        auto rvec = vec3(radius, radius, radius);
        aabb box1(center.at(0) - rvec, center.at(0) + rvec);
        aabb box2(center.at(1) - rvec, center.at(1) + rvec);
        bbox = aabb(box1, box2);
    }

    aabb bounding_box() const override {
        return bbox;
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override;

private:
    ray                       center;
    double                    radius;
    std::shared_ptr<material> mat;
    aabb                      bbox;
};


#endif // SPHERE_H
