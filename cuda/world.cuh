//
// Created by amuhak on 12/31/2025.
//

#ifndef RAYTRACING_WORLD_CUH
#define RAYTRACING_WORLD_CUH

#include "hittable.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"

#include <curand_kernel.h>

__global__ void create_world(hittable **d_world, curandState *rand_state) {
    // Only one thread initializes the world to avoid race conditions/duplicates
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Create a local copy of the random state for performance
        curandState local_rand_state = *rand_state;

        // Make the list
        auto tmp_list = new hittable_list();

        // 1. Ground Material and Sphere
        auto material_ground = new lambertian(vec3(0.5f, 0.5f, 0.5f));
        tmp_list->add(new sphere(vec3(0, -1000, 0), 1000, material_ground));

        // 2. Random Small Spheres
        for (int a = -20; a < 20; a++) {
            for (int b = -20; b < 20; b++) {
                float choose_mat = curand_uniform(&local_rand_state);

                // Random center position
                float r1 = curand_uniform(&local_rand_state);
                float r2 = curand_uniform(&local_rand_state);
                vec3  center(a + 0.9f * r1, 0.2f, b + 0.9f * r2);

                if ((center - vec3(4, 0.2f, 0)).length() > 0.9f) {
                    material *sphere_material;

                    if (choose_mat < 0.8f) {
                        // diffuse: albedo = random * random (to approximate gamma)
                        vec3 rand_vec1 = vec3(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state),
                                              curand_uniform(&local_rand_state));
                        vec3 rand_vec2 = vec3(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state),
                                              curand_uniform(&local_rand_state));
                        vec3 albedo    = rand_vec1 * rand_vec2;

                        sphere_material = new lambertian(albedo);
                        tmp_list->add(new sphere(center, 0.2f, sphere_material));
                    } else if (choose_mat < 0.95f) {
                        // metal: albedo random(0.5, 1), fuzz random(0, 0.5)
                        vec3  albedo = vec3(0.5f * (1.0f + curand_uniform(&local_rand_state)),
                                            0.5f * (1.0f + curand_uniform(&local_rand_state)),
                                            0.5f * (1.0f + curand_uniform(&local_rand_state)));
                        float fuzz   = 0.5f * curand_uniform(&local_rand_state);

                        sphere_material = new metal(albedo, fuzz);
                        tmp_list->add(new sphere(center, 0.2f, sphere_material));
                    } else {
                        // glass
                        sphere_material = new dielectric(1.5f);
                        tmp_list->add(new sphere(center, 0.2f, sphere_material));
                    }
                }
            }
        }

        // 3. The Three Big Spheres
        auto material1 = new dielectric(1.5f);
        tmp_list->add(new sphere(vec3(0, 1, 0), 1.0f, material1));

        auto material2 = new lambertian(vec3(0.4f, 0.2f, 0.1f));
        tmp_list->add(new sphere(vec3(-4, 1, 0), 1.0f, material2));

        auto material3 = new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f);
        tmp_list->add(new sphere(vec3(4, 1, 0), 1.0f, material3));

        // Set the world pointer
        *d_world = tmp_list;

        // Save the RNG state back (optional, but good practice)
        *rand_state = local_rand_state;
    }
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
