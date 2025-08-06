//
// Created by amuhak on 8/2/2025.
//

#ifndef PRETTYPRINT_H
#define PRETTYPRINT_H

// Add this to the top of main.cpp
#include <iostream>
#include <chrono>
#include <format>

#include "grid.hpp"

// --- Platform-specific headers ---
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__)
#include <sys/ioctl.h> // For ioctl, TIOCGWINSZ, and struct winsize
#include <unistd.h>    // For STDOUT_FILENO
#endif // _WIN32 || __linux__

// ... the rest of your code, including the get_terminal_size function ...

/**
 * @brief Retrieves the current width and height of the terminal window.
 * * This function is cross-platform and uses the appropriate system calls for
 * either Windows or Linux to determine the console dimensions.
 * * @param width Reference to an integer to store the terminal width (columns).
 * @param height Reference to an integer to store the terminal height (rows).
 */


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