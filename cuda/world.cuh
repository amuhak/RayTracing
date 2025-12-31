//
// Created by amuhak on 12/31/2025.
//

#ifndef RAYTRACING_WORLD_CUH
#define RAYTRACING_WORLD_CUH

#include "hittable.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"

__global__ void create_world(hittable **d_world) {
    if (!(threadIdx.x == 0 && blockIdx.x == 0)) {
        return;
    }

    // Make the list
    const auto tmp_list = new hittable_list();

    // Define materials
    auto material_ground = new lambertian(vec3(0.8, 0.8, 0.0));
    auto material_center = new lambertian(vec3(0.1, 0.2, 0.5));
    auto material_left   = new metal(vec3(0.8, 0.8, 0.8), 0.3f);
    auto material_right  = new metal(vec3(0.8, 0.6, 0.2), 1.0f);

    // Add spheres to the list
    tmp_list->add(new sphere(vec3(0, -100.5, -1), 100, material_ground));
    tmp_list->add(new sphere(vec3(0, 0, -1.2), 0.5, material_center));
    tmp_list->add(new sphere(vec3(-1, 0, -1), 0.5, material_left));
    tmp_list->add(new sphere(vec3(1, 0, -1), 0.5, material_right));

    // Set the world pointer (return)
    *d_world = tmp_list;
}

__global__ void cleanup_world(hittable **d_world) {
    if (!(threadIdx.x == 0 && blockIdx.x == 0)) {
        return;
    }
    if (!d_world || !*d_world) {
        printf("Error: Invalid world pointer in cleanup\n");
        return;
    }
    delete *d_world;
    *d_world = nullptr;
}

#endif // RAYTRACING_WORLD_CUH
