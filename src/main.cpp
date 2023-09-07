#include "network.hpp"

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <cstdlib>
#include <iostream>
#include <utility>

namespace
{

template <typename F>
class Scope_guard
{
public:
    explicit Scope_guard(F &&f) : m_f {std::forward<F>(f)}
    {
    }

    ~Scope_guard() noexcept
    {
        m_f();
    }

    Scope_guard(const Scope_guard &) = delete;
    Scope_guard(Scope_guard &&) noexcept = delete;
    Scope_guard &operator=(const Scope_guard &) = delete;
    Scope_guard &operator=(Scope_guard &&) noexcept = delete;

private:
    F m_f;
};

template <typename F>
Scope_guard(F &&) -> Scope_guard<F>;

} // namespace

int main(int argc, char *argv[])
{
    std::cout << "Args: ";
    for (int i {}; i < argc; ++i)
    {
        std::cout << argv[i] << ' ';
    }
    std::cout << '\n';

    Network network {};
    network_init(network, {10, 10});

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0)
    {
        std::cerr << SDL_GetError() << '\n';
        return EXIT_FAILURE;
    }
    const Scope_guard sdl_guard([] { SDL_Quit(); });

    const auto window = SDL_CreateWindow("Neural Image",
                                         SDL_WINDOWPOS_UNDEFINED,
                                         SDL_WINDOWPOS_UNDEFINED,
                                         1280,
                                         720,
                                         {});
    if (window == nullptr)
    {
        std::cerr << SDL_GetError() << '\n';
        return EXIT_FAILURE;
    }
    const Scope_guard window_guard([window] { SDL_DestroyWindow(window); });

    const auto renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == nullptr)
    {
        std::cerr << SDL_GetError() << '\n';
        return EXIT_FAILURE;
    }
    const Scope_guard renderer_guard([renderer]
                                     { SDL_DestroyRenderer(renderer); });

    const int texture_width {128};
    const int texture_height {72};
    const auto texture = SDL_CreateTexture(renderer,
                                           SDL_PIXELFORMAT_ABGR8888,
                                           SDL_TEXTUREACCESS_STREAMING,
                                           texture_width,
                                           texture_height);
    if (texture == nullptr)
    {
        std::cerr << SDL_GetError() << '\n';
        return EXIT_FAILURE;
    }
    const Scope_guard texture_guard([texture] { SDL_DestroyTexture(texture); });

    bool quit {false};
    while (!quit)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
            {
                quit = true;
            }
        }

        Uint8 *pixels;
        int pitch;
        if (SDL_LockTexture(
                texture, nullptr, reinterpret_cast<void **>(&pixels), &pitch) !=
            0)
        {
            std::cerr << SDL_GetError() << '\n';
            return EXIT_FAILURE;
        }

        for (int i {0}; i < texture_height; ++i)
        {
            for (int j {0}; j < texture_width; ++j)
            {
                const auto pixel_index = i * texture_width + j;
                const auto x = static_cast<float>(j) /
                                   static_cast<float>(texture_width - 1) *
                                   2.0f -
                               1.0f;
                const auto y = static_cast<float>(i) /
                                   static_cast<float>(texture_height - 1) *
                                   2.0f -
                               1.0f;
                const auto output_to_u8 = [](float output)
                { return static_cast<Uint8>((output + 1.0f) * 0.5f * 255.0f); };
                const auto prediction = network_predict(network, x, y);
                pixels[pixel_index * 4 + 0] = output_to_u8(prediction[0]);
                pixels[pixel_index * 4 + 1] = output_to_u8(prediction[1]);
                pixels[pixel_index * 4 + 2] = output_to_u8(prediction[2]);
                pixels[pixel_index * 4 + 3] = SDL_ALPHA_OPAQUE;
            }
        }

        SDL_UnlockTexture(texture);

        if (SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE) != 0)
        {
            std::cerr << SDL_GetError() << '\n';
            return EXIT_FAILURE;
        }

        if (SDL_RenderClear(renderer) != 0)
        {
            std::cerr << SDL_GetError() << '\n';
            return EXIT_FAILURE;
        }

        if (SDL_RenderCopy(renderer, texture, nullptr, nullptr) != 0)
        {
            std::cerr << SDL_GetError() << '\n';
            return EXIT_FAILURE;
        }

        SDL_RenderPresent(renderer);
    }

    return EXIT_SUCCESS;
}
