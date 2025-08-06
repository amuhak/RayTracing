//
// Created by amuhak on 8/2/2025.
//

#include "prettyPrint.hpp"
#include <thread>
#include "grid.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

void prettyPrint::get_terminal_size(uint32_t &width, uint32_t &height) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    width  = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    height = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
#elif defined(__linux__)
    struct winsize w;
    // ioctl is a low-level system call to manipulate device parameters.
    // TIOCGWINSZ is the specific request to "get window size".
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    width  = static_cast<uint32_t>(w.ws_col);
    height = static_cast<uint32_t>(w.ws_row);
#else
    // Default values for other or unknown operating systems
    width  = 80;
    height = 24;
#endif // _WIN32 || __linux__
    // Make sure the values aren't crazy:
    if (width > 1000 || height > 500) {
        width  = 80;
        height = 24;
    }
}

constexpr auto CLEAR_LINE                      = "\x1B[2K";
uint64_t       prettyPrint::lastPixelsRendered = 0;
constexpr auto SLEEP_DURATION                  = std::chrono::milliseconds(100);

void prettyPrint::update(const size_t done, const size_t total) {
    // Get size
    uint32_t width, height;
    get_terminal_size(width, height);

    const double   percentDone{static_cast<double>(done) / static_cast<double>(total)};
    const uint32_t barWidth(width - 2); // Leave space for the brackets
    const auto     hashes = static_cast<size_t>(percentDone * static_cast<double>(barWidth));
    const size_t   spaces = barWidth - hashes;

    const uint64_t change_inPixels = done - lastPixelsRendered;
    lastPixelsRendered             = done;
    using ns                       = std::chrono::duration<double, std::nano>;

    // Cast SLEEP_DURATION to our new floating-point type BEFORE dividing.
    const auto time_to_render_one_pixel = ns(SLEEP_DURATION) / change_inPixels;

    // Now, time_left will also be of type float_ms and will have the correct value.
    auto time_left = (total - done) * time_to_render_one_pixel;

    const auto hours = std::chrono::duration_cast<std::chrono::hours>(time_left);

    // 2. Subtract the hours part to get the remainder
    time_left -= hours;

    // 3. From the remainder, calculate the total number of integer minutes
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(time_left);
    time_left -= minutes;

    // 4. From the new remainder, calculate the integer seconds
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_left);
    time_left -= seconds;

    // 5. The final remainder is the fractional part, which we cast to milliseconds
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time_left);

    std::string timeLeftStr = std::format("{:02}:{:02}:{:02}.{:03}", hours.count(), minutes.count(), seconds.count(),
                                          milliseconds.count());


    // Clear line and print status
    std::cout << CLEAR_LINE << "Rendering: " << std::format("{:.2f}", percentDone * 100) << "% Done (" << done << "/"
              << total << " Pixels Rendered) Time Left: " << timeLeftStr;


    // Move to next line and print progress bar
    std::cout << "\n"
              << CLEAR_LINE << "[" << std::string(hashes, '#') << std::string(spaces, ' ') << "] " << std::flush;

    // Move cursor back up to status line for next iteration
    if (done < total) {
        std::cout << "\x1B[2A"; // Move cursor up 2 lines to beginning of status line
    }
}


void prettyPrint::run() const {
    while (keepUpdating) {
        std::this_thread::sleep_for(SLEEP_DURATION);
        update(image->total_done.load(std::memory_order_relaxed), image->size);
    }
    update(image->size, image->size);
}
