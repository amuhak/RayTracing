//
// Created by amuhak on 8/3/2025.
//
#include "display.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/System/Vector2.hpp>
#include <thread>
#include <vector>


void display::run(uint32_t width, uint32_t height, const std::vector<uint8_t> &data) {
    sf::RenderWindow window;
    sf::Texture      texture(sf::Vector2{width, height});
    window.create(sf::VideoMode({width, height}), "Ray Tracer Display");
    window.setFramerateLimit(0);
    while (keepUpdating) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (data.empty()) {
            continue; // No data to display
        }
        while (const std::optional event = window.pollEvent()) {
            if (event.has_value() && event->is<sf::Event::Closed>()) {
                keepUpdating = false;
            }
        }
        texture.update(data.data());
        sf::Sprite sprite(texture);
        window.clear();
        window.draw(sprite);
        window.display();
    }
    window.close();
}
