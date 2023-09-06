#include "network.hpp"

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[])
{
    std::cout << "Args: ";
    for (int i {}; i < argc; ++i)
    {
        std::cout << argv[i] << ' ';
    }
    std::cout << '\n';

    const auto network = create_network({10, 16, 10});

    const auto output = predict(network, 0.5, 0.5);

    std::cout << output << '\n';

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0)
    {
        std::cerr << SDL_GetError() << '\n';
        return EXIT_FAILURE;
    }

    SDL_Quit();

    return EXIT_SUCCESS;
}
