//
// Created by amuhak on 4/7/2025.
//

#include "rtweekend.hpp"
#include <thread>
#include <random>
#include <bit>

thread_local static std::mt19937_64 gen{
    std::hash<std::thread::id>{}(std::this_thread::get_id())
};


thread_local static uint64_t xState{gen()}, yState{gen()}; // set to nonzero seed

uint64_t romuDuoJr_random() {
    const uint64_t xp = xState;
    xState = 15241094284759029579u * yState;
    yState = yState - xp;
    yState = std::rotl(yState, 27);
    return xp;
}

double degrees_to_radians(const double degrees) {
    return degrees * pi / 180.0;
}

/**
 * @return Random real in [0,1)
 */
double random_double() {
    const uint64_t rand_int = romuDuoJr_random();

    // 2. The bit pattern for 1.0 in double precision is 0x3FF0000000000000.
    //    We take the top 52 bits of our random integer for the mantissa.
    //    (rand_int >> 12) gets the top 52 bits.
    const uint64_t double_bits = 0x3FF0000000000000 | (rand_int >> 12);

    // 3. Reinterpret the bits as a double and subtract 1.0 to map to [0, 1).
    return std::bit_cast<double>(double_bits) - 1.0;
}

/**
 *
 * @return A random double in the range [-1, 1].
 */
double random_unit_double() {
    return random_double() * 2.0 - 1.0;
}

double random_double(const double min, const double max) {
    std::uniform_real_distribution dis2(min, max);
    return dis2(gen);
}
