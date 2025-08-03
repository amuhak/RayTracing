//
// Created by amuhak on 8/2/2025.
//

#include "prettyPrint.h"
#include <thread>

void prettyPrint::get_terminal_size(uint32_t &width, uint32_t &height) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    width = static_cast<int>(csbi.srWindow.Right - csbi.srWindow.Left + 1);
    height = static_cast<int>(csbi.srWindow.Bottom - csbi.srWindow.Top + 1);
#elif defined(__linux__)
    struct winsize w;
    // ioctl is a low-level system call to manipulate device parameters.
    // TIOCGWINSZ is the specific request to "get window size".
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    width = static_cast<int>(w.ws_col);
    height = static_cast<int>(w.ws_row);
#else
    // Default values for other or unknown operating systems
    width = 80;
    height = 24;
#endif // _WIN32 || __linux__
    // Make sure the values arent crazy:
    if (width > 1000 || height > 500) {
        width = 80;
        height = 24;
    }
}

constexpr auto CLEAR_LINE = "\x1B[2K";
constexpr auto CURSOR_UP_2 = "\x1B[2A";

void prettyPrint::update(const size_t done, const size_t total) {
    // Get size
    uint32_t width, height;
    get_terminal_size(width, height);

    const double percentDone{static_cast<double>(done) / static_cast<double>(total)};
    const uint32_t barWidth(width - 2); // Leave space for the brackets
    const auto hashes = static_cast<size_t>(percentDone * static_cast<double>(barWidth));
    const size_t spaces = barWidth - hashes;

    // Clear line and print status
    std::cout << CLEAR_LINE
            << "Rendering: "
            << std::format("{:.2f}", percentDone * 100)
            << "% Done ("
            << done << "/" << total
            << "Pixels Rendered)";

    // Move to next line and print progress bar
    std::cout << "\n"
            << CLEAR_LINE
            << "["
            << std::string(hashes, '#')
            << std::string(spaces, ' ')
            << "] "
            << std::flush;

    // Move cursor back up to status line for next iteration
    if (done < total) {
        std::cout << "\x1B[2A"; // Move cursor up 2 lines to beginning of status line
    }
}

void prettyPrint::run() {
    while (keepUpdating) {
        // Wait for 10ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // Update the status
        update(image->total_done.load(std::memory_order_relaxed), image->size);
    }
}
