//
// Created by amuhak on 8/2/2025.
//

#ifndef PRETTYPRINT_H
#define PRETTYPRINT_H

#include <chrono>
//#include "grid.hpp"

class grid;

class prettyPrint {
    static constexpr auto tenMs = std::chrono::time_point<std::chrono::high_resolution_clock>(
        std::chrono::milliseconds(10)
    );
    std::chrono::time_point<std::chrono::high_resolution_clock> timeOfLastUpdate{};
    const grid *image;

    static uint64_t lastPixelsRendered;

    static void get_terminal_size(uint32_t &width, uint32_t &height);

    static void update(size_t done, size_t total);

public:
    explicit prettyPrint(const grid &img) {
        image = &img;
        // Initialize the time of last update to the current time
        timeOfLastUpdate = std::chrono::high_resolution_clock::now();
    }

    bool keepUpdating{true};

    void run() const;
};


#endif //PRETTYPRINT_H
