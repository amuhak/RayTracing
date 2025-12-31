//
// Created by amuhak on 12/31/2025.
//

#ifndef RAYTRACING_MATERIAL_CUH
#define RAYTRACING_MATERIAL_CUH

#include "hittable.cuh"

class material {
public:
    virtual ~material() = default;

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered,
                                    curandState &rand_state) const {
        return false;
    }
};

class lambertian : public material {
public:
    __device__ lambertian(const vec3 &albedo) : albedo(albedo) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered,
                            curandState &rand_state) const override {
        auto scatter_direction = rec.normal + random_unit_vector(rand_state);

        if (scatter_direction.near_zero()) {
            scatter_direction = rec.normal;
        }

        scattered   = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    vec3 albedo;
};

class metal : public material {
public:
    __device__ metal(const vec3 &albedo, const float fuzz) : albedo(albedo), fuzz(min(1.0f, fuzz)) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered,
                            curandState &rand_state) const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected      = unit_vector(reflected) + (fuzz * random_unit_vector(rand_state));
        scattered      = ray(rec.p, reflected);
        attenuation    = albedo;
        return dot(scattered.direction(), rec.normal) > 0;
    }

private:
    vec3  albedo;
    float fuzz;
};

#endif // RAYTRACING_MATERIAL_CUH
