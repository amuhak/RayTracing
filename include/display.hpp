//
// Created by amuhak on 8/3/2025.
//

#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include <cstdint>
#include <vector>

class display {
public:
    bool keepUpdating{true};

    void run(uint32_t width, uint32_t height, const std::vector<uint8_t> &data);
};


#endif
