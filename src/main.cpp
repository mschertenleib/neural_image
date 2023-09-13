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

#define CONCATENATE_IMPL(a, b) a##b
#define CONCATENATE(a, b)      CONCATENATE_IMPL(a, b)
#define DEFER(f)               const Scope_guard CONCATENATE(scope_guard_, __LINE__)(f)

inline void sdl_check(int result)
{
    if (result != 0)
    {
        throw std::runtime_error(SDL_GetError());
    }
}

inline void sdl_check(const auto *pointer)
{
    if (pointer == nullptr)
    {
        throw std::runtime_error(SDL_GetError());
    }
}

void application_main(Network &network)
{
    sdl_check(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS));
    DEFER([] { SDL_Quit(); });

    const auto window = SDL_CreateWindow("Neural Image",
                                         SDL_WINDOWPOS_UNDEFINED,
                                         SDL_WINDOWPOS_UNDEFINED,
                                         1280,
                                         720,
                                         {});
    sdl_check(window);
    DEFER([window] { SDL_DestroyWindow(window); });

    const auto renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    sdl_check(renderer);
    DEFER([renderer] { SDL_DestroyRenderer(renderer); });

    const int texture_width {21};
    const int texture_height {21};
    const auto texture = SDL_CreateTexture(renderer,
                                           SDL_PIXELFORMAT_ABGR8888,
                                           SDL_TEXTUREACCESS_STREAMING,
                                           texture_width,
                                           texture_height);
    sdl_check(texture);
    DEFER([texture] { SDL_DestroyTexture(texture); });

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
        sdl_check(SDL_LockTexture(
            texture, nullptr, reinterpret_cast<void **>(&pixels), &pitch));

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
                constexpr auto output_to_u8 = [](float output)
                { return static_cast<Uint8>((output + 1.0f) * 0.5f * 255.0f); };
                network_predict(network, {x, y});
                const auto prediction = network.output_layer.activations;
                pixels[pixel_index * 4 + 0] = output_to_u8(prediction[0]);
                pixels[pixel_index * 4 + 1] = output_to_u8(prediction[1]);
                pixels[pixel_index * 4 + 2] = output_to_u8(prediction[2]);
                pixels[pixel_index * 4 + 3] = SDL_ALPHA_OPAQUE;
            }
        }

        SDL_UnlockTexture(texture);
        sdl_check(SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE));
        sdl_check(SDL_RenderClear(renderer));
        sdl_check(SDL_RenderCopy(renderer, texture, nullptr, nullptr));
        SDL_RenderPresent(renderer);

        constexpr int image[7][7] {{0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 1, 1, 1, 1, 0},
                                   {0, 0, 0, 1, 0, 0, 0},
                                   {0, 0, 0, 1, 1, 0, 0},
                                   {0, 0, 1, 0, 0, 0, 0},
                                   {0, 0, 1, 0, 0, 1, 0},
                                   {0, 1, 0, 0, 0, 0, 0}};

        for (int i {0}; i < 7; ++i)
        {
            for (int j {0}; j < 7; ++j)
            {
                const auto x =
                    static_cast<float>(j) / static_cast<float>(7 - 1) * 2.0f -
                    1.0f;
                const auto y =
                    static_cast<float>(i) / static_cast<float>(7 - 1) * 2.0f -
                    1.0f;
                const float learning_rate {0.01f};
                stochastic_gradient_descent(network,
                                            Eigen::Vector2f(x, y),
                                            image[i][j]
                                                ? Eigen::Vector3f(1, 1, 1)
                                                : Eigen::Vector3f(-1, -1, -1),
                                            learning_rate);
            }
        }
    }
}

} // namespace

int main()
{
    try
    {
        Network network {};
        network_init(network, {20, 20, 20, 20});

        Eigen::internal::set_is_malloc_allowed(false);

        application_main(network);

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "Unknown exception thrown\n";
        return EXIT_FAILURE;
    }
}
