//
// Created by amuhak on 4/7/2025.
//

#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <limits>
#include <cstdint>

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

// Romu Pseudorandom Number Generators
//
// Copyright 2020 Mark A. Overton
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ------------------------------------------------------------------------------------------------
//
// Website: romu-random.org
// Paper:   http://arxiv.org/abs/2002.11331
uint64_t romuDuoJr_random();

double degrees_to_radians(double degrees);

/**
 * @return Random real in [0,1)
 */
double random_double();

/**
 * @return A random double in the range [-1, 1].
 */
double random_unit_double();

/**
 * @param min minimum value
 * @param max maximum value
 * @return A random double in the range [min, max].
 */
double random_double(double min, double max);

#endif //RTWEEKEND_H