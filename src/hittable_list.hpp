//
// Created by amuhak on 4/7/2025.
//

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.hpp"

#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

class hittable_list : public hittable {
public:
    std::vector<shared_ptr<hittable> > objects;

    hittable_list() = default;

    explicit hittable_list(const shared_ptr<hittable> &object) { add(object); }

    void clear() { objects.clear(); }

    void add(const shared_ptr<hittable> &object) {
        objects.push_back(object);
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto &object: objects) {
            // Using closest_so_far as the maximum t value prevents overlapping hits.
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};


#endif //HITTABLE_LIST_H
