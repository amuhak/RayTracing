//
// Created by amuhak on 12/31/2025.
//

#ifndef RAYTRACING_MATERIAL_CUH
#define RAYTRACING_MATERIAL_CUH

#include "hittable.cuh"
#include "vec3.cuh"

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

class dielectric : public material {
public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered,
                            curandState &rand_state) const override {
        attenuation = vec3(1.0, 1.0, 1.0);
        float ri    = rec.front_face ? 1.0f / refraction_index : refraction_index;

        vec3  unit_direction = unit_vector(r_in.direction());
        float cos_theta      = std::fminf(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta      = std::sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > vec3::random_float(rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refraction_index;

    __device__ static float reflectance(const float cosine, const float refraction_index) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1 - refraction_index) / (1 + refraction_index);
        r0       = r0 * r0;
        return r0 + (1 - r0) * std::powf(1 - cosine, 5);
    }
};

#endif // RAYTRACING_MATERIAL_CUH
