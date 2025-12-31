//
// Created by amuhak on 12/30/2025.
//

#ifndef RAYTRACING_HITTABLE_LIST_CUH
#define RAYTRACING_HITTABLE_LIST_CUH

#include "cudaTools.cuh"
#include "hittable.cuh"

class hittable_list : public hittable {
public:
    hittable **d_objects{nullptr};
    size_t     size{};

    hittable_list() = default;

    __device__ explicit hittable_list(hittable *object) {
        add(object);
    }

    __device__ void clear() {
        for (size_t i = 0; i < size; i++) {
            delete d_objects[i];
        }
        delete[] d_objects;
        d_objects = nullptr;
        size      = 0;
    }

    /**
     * We assume ownership of the object
     * @param object the obj to add
     */
    __device__ void add(hittable *object) {
        if (!(threadIdx.x == 0 && blockIdx.x == 0)) {
            return;
        }
        const auto d_old = d_objects;
        d_objects        = new hittable *[size + 1];
        for (size_t i = 0; i < size; i++) {
            d_objects[i] = d_old[i];
        }
        d_objects[size] = object;
        delete[] d_old;
        size++;
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        hit_record temp_rec;
        bool       hit_anything   = false;
        auto       closest_so_far = ray_t.max;

        for (size_t i = 0; i < size; i++) {
            hittable *object = d_objects[i];
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything   = true;
                closest_so_far = temp_rec.t;
                rec            = temp_rec;
            }
        }

        return hit_anything;
    }

    /**
     * Make sure to call clear() before destroying the list
     */
    __device__ ~hittable_list() override {
        clear();
    }
};


#endif // RAYTRACING_HITTABLE_LIST_CUH
