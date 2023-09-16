#include "network.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

namespace
{

struct Image
{
    std::size_t width;
    std::size_t height;
    std::vector<std::uint8_t> pixel_data;
};

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

[[maybe_unused]] [[nodiscard]] inline float loss(const Network &network,
                                                 const Eigen::Vector3f &output)
{
    return 0.5f *
           (output - network.output_layer.activations).array().square().sum();
}

[[nodiscard]] constexpr float u8_to_float(std::uint8_t u)
{
    return static_cast<float>(u) / 255.0f;
}

[[nodiscard]] constexpr std::uint8_t float_to_u8(float f)
{
    return static_cast<std::uint8_t>(f * 255.0f);
}

[[nodiscard]] Image load_image(const char *file_name)
{
    int width;
    int height;
    int channels_in_file;
    constexpr int desired_channels {3};
    auto *image_data = stbi_load(
        file_name, &width, &height, &channels_in_file, desired_channels);
    if (image_data == nullptr)
    {
        throw std::runtime_error(stbi_failure_reason());
    }
    DEFER([image_data] { stbi_image_free(image_data); });

    Image image {.width = static_cast<std::size_t>(width),
                 .height = static_cast<std::size_t>(height),
                 .pixel_data = {}};

    image.pixel_data.assign(
        image_data,
        image_data +
            static_cast<std::size_t>(width * height * desired_channels));

    return image;
}

[[nodiscard]] inline Eigen::Vector3f
get_pixel(const Image &image, std::size_t i, std::size_t j)
{
    const auto pixel_index = i * image.width + j;
    return {u8_to_float(image.pixel_data[pixel_index * 3 + 0]),
            u8_to_float(image.pixel_data[pixel_index * 3 + 1]),
            u8_to_float(image.pixel_data[pixel_index * 3 + 2])};
}

void create_test_image(const char *file_name)
{
    constexpr std::uint8_t pixels[7][13] {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 255, 0, 255, 0, 255, 255, 255, 0, 255, 0, 255, 0},
        {0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 0, 255, 0},
        {0, 255, 255, 255, 0, 255, 255, 0, 0, 0, 255, 0, 0},
        {0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 255, 0, 0},
        {0, 255, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    constexpr std::size_t height {std::size(pixels)};
    constexpr std::size_t width {std::size(pixels[0])};

    std::ofstream file(file_name);
    file << "P5 " << width << ' ' << height << " 255 ";
    file.write(reinterpret_cast<const char *>(pixels), width * height);
}

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

void application_main(Network &network, Eigen::VectorXf &input)
{
    sdl_check(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS));
    DEFER([] { SDL_Quit(); });

    const auto window = SDL_CreateWindow("Neural Image",
                                         SDL_WINDOWPOS_UNDEFINED,
                                         SDL_WINDOWPOS_UNDEFINED,
                                         1280,
                                         720,
                                         SDL_WINDOW_RESIZABLE);
    sdl_check(window);
    DEFER([window] { SDL_DestroyWindow(window); });

    const auto renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    sdl_check(renderer);
    DEFER([renderer] { SDL_DestroyRenderer(renderer); });

    const int texture_width {130};
    const int texture_height {70};
    const auto texture = SDL_CreateTexture(renderer,
                                           SDL_PIXELFORMAT_ABGR8888,
                                           SDL_TEXTUREACCESS_STREAMING,
                                           texture_width,
                                           texture_height);
    sdl_check(texture);
    DEFER([texture] { SDL_DestroyTexture(texture); });

    constexpr auto image_file_name {"hey.pgm"};
    create_test_image(image_file_name);
    const auto image = load_image(image_file_name);

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
                input.setZero();
                input << x, y;
                predict(network, input);
                const auto &prediction = network.output_layer.activations;
                pixels[pixel_index * 4 + 0] =
                    float_to_u8((prediction[0] + 1.0f) * 0.5f);
                pixels[pixel_index * 4 + 1] =
                    float_to_u8((prediction[1] + 1.0f) * 0.5f);
                pixels[pixel_index * 4 + 2] =
                    float_to_u8((prediction[2] + 1.0f) * 0.5f);
                pixels[pixel_index * 4 + 3] = SDL_ALPHA_OPAQUE;
            }
        }

        SDL_UnlockTexture(texture);
        sdl_check(SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE));
        sdl_check(SDL_RenderClear(renderer));
        sdl_check(SDL_RenderCopy(renderer, texture, nullptr, nullptr));
        SDL_RenderPresent(renderer);

        float total_loss {0.0f};
        const float learning_rate {0.005f};
        for (std::size_t i {0}; i < image.height; ++i)
        {
            for (std::size_t j {0}; j < image.width; ++j)
            {
                const auto x = static_cast<float>(j) /
                                   static_cast<float>(image.width - 1) * 2.0f -
                               1.0f;
                const auto y = static_cast<float>(i) /
                                   static_cast<float>(image.height - 1) * 2.0f -
                               1.0f;
                input.setZero();
                input << x, y;
                const auto output = get_pixel(image, i, j);
                stochastic_gradient_descent(
                    network, input, output, learning_rate);
                total_loss += loss(network, output);
            }
        }
        std::cout << total_loss << '\n';
    }
}

} // namespace

int main()
{
    try
    {
        Network network {};
        init_network(network, {2, 40, 20});
        Eigen::VectorXf input(2);

        Eigen::internal::set_is_malloc_allowed(false);

        application_main(network, input);

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
