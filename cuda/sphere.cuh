//
// Created by amuhak on 12/30/2025.
//

#ifndef RAYTRACING_SPHERE_CUH
#define RAYTRACING_SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"

class sphere : public hittable {
public:
    __device__ sphere(const point3 &center, const float radius) : center(center), radius(std::fmax(0.0f, radius)) {
    }

    __device__ bool hit(const ray &r, const float ray_tmin, const float ray_tmax, hit_record &rec) const override {
        vec3 oc = center - r.origin();
        auto a  = r.direction().length_squared();
        auto h  = dot(r.direction(), oc);
        auto c  = oc.length_squared() - radius * radius;

        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t               = root;
        rec.p               = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);


        return true;
    }

private:
    point3 center;
    float  radius;
};


#endif // RAYTRACING_SPHERE_CUH
