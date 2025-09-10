//
// Created by amuhak on 9/9/2025.
//

#include "aabb.hpp"
const aabb aabb::empty    = aabb(interval::empty, interval::empty, interval::empty);
const aabb aabb::universe = aabb(interval::universe, interval::universe, interval::universe);
