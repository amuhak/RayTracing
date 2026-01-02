//
// Created by amuhak on 12/31/2025.
//

#ifndef RAYTRACING_CAMERA_CUH
#define RAYTRACING_CAMERA_CUH

#include "hittable.cuh"
#include "interval.cuh"
#include "material.cuh"
#include "ray.cuh"

constexpr int    samples_per_pixel   = 100;
constexpr float  pixel_samples_scale = 1.0f / static_cast<float>(samples_per_pixel);
constexpr int    max_depth           = 50;
constexpr float  vfov                = 20;
constexpr point3 lookfrom{-2, 2, 1};
constexpr point3 lookat{0, 0, -1};
constexpr vec3   vup{0, 1, 0}; // Camera-relative "up" direction


__device__ vec3 color(const ray &r, const hittable *const *d_world, curandState &rand_state) {
    const hittable &world  = **d_world;
    ray             r_copy = r;
    hit_record      rec;
    vec3            ans{1, 1, 1};
    int             depth{};

    for (; depth < max_depth; depth++) {
        if (!world.hit(r_copy, interval(0.001f, infinity), rec)) {
            break;
        }
        ray  scattered;
        vec3 attenuation;
        if (rec.mat->scatter(r_copy, rec, attenuation, scattered, rand_state)) {
            r_copy = scattered;
            ans    = ans * attenuation;
        } else {
            return {0, 0, 0};
        }
    }

    if (depth == max_depth) {
        return {0, 0, 0};
    }

    const vec3  unit_direction = unit_vector(r_copy.direction());
    const float a              = 0.5f * (unit_direction.y() + 1.0f);
    return ans * ((1.0f - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0));
}

__device__ ray get_ray(const uint32_t i, const uint32_t j, const point3 pixel00_loc, const vec3 pixel_delta_u,
                       const vec3 pixel_delta_v, const point3 camera_center, curandState &rand_state) {

    vec3 offset{curand_uniform(&rand_state) - 0.5f, curand_uniform(&rand_state) - 0.5f, 0};
    auto pixel_sample = pixel00_loc + (i + offset.x()) * pixel_delta_u + (j + offset.y()) * pixel_delta_v;

    auto ray_origin    = camera_center;
    auto ray_direction = pixel_sample - ray_origin;

    return {ray_origin, ray_direction};
}


__global__ void render(float4 *fb, const int max_x, const int max_y, const point3 pixel00_loc, const vec3 pixel_delta_u,
                       const vec3 pixel_delta_v, const point3 camera_center, const hittable *const *d_world,
                       curandState *d_rand_state) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) {
        return;
    }
    const uint32_t pixel_index = j * max_x + i;
    curandState   &rand_state  = d_rand_state[pixel_index];
    vec3           pixel_color(0, 0, 0);
    for (int sample = 0; sample < samples_per_pixel; sample++) {
        ray r = get_ray(i, j, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center, rand_state);
        pixel_color += color(r, d_world, rand_state);
    }
    const auto ans = pixel_samples_scale * pixel_color;

    fb[pixel_index] = make_float4(ans.e[0], ans.e[1], ans.e[2], 1.0f);
}


#endif // RAYTRACING_CAMERA_CUH
