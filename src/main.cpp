#include "network.hpp"

#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

namespace
{

constexpr int input_size {256};

struct Training_pair
{
    Eigen::Vector<float, input_size> input;
    Eigen::Vector3f output;
};

struct Training_dataset
{
    std::size_t image_width;
    std::size_t image_height;
    std::vector<Training_pair> training_pairs;
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
                                                 const Network::Output &output)
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

void get_fourier_features_positional_encoding(
    Eigen::Vector<float, input_size> &v,
    std::size_t max_image_dimension,
    const Eigen::Vector2f &coordinates)
{
    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;
    const auto max_frequency = static_cast<float>(max_image_dimension) * 0.5f;
    constexpr int m {input_size / 4};
    for (int j {0}; j < m; ++j)
    {
        const auto frequency = std::pow(
            max_frequency, static_cast<float>(j) / static_cast<float>(m - 1));
        v(4 * j + 0) = std::cos(two_pi * frequency * coordinates.x());
        v(4 * j + 1) = std::cos(two_pi * frequency * coordinates.y());
        v(4 * j + 2) = std::sin(two_pi * frequency * coordinates.x());
        v(4 * j + 3) = std::sin(two_pi * frequency * coordinates.y());
    }
}

[[nodiscard]] Training_dataset load_image(const char *file_name)
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

    Training_dataset dataset {.image_width = static_cast<std::size_t>(width),
                              .image_height = static_cast<std::size_t>(height),
                              .training_pairs = {}};
    dataset.training_pairs.resize(dataset.image_width * dataset.image_height);

    std::random_device rd {};
    std::minstd_rand rng(rd());
    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;
    const auto std_dev = static_cast<float>(std::max(dataset.image_width,
                                                     dataset.image_height)) *
                         0.1f;
    std::normal_distribution<float> distribution(0.0f, std_dev);
    const auto generate_weight = [&](float) { return distribution(rng); };
    Eigen::Matrix<float, input_size / 2, 2> frequencies;
    frequencies = frequencies.unaryExpr(generate_weight);

    for (std::size_t i {0}; i < dataset.image_height; ++i)
    {
        for (std::size_t j {0}; j < dataset.image_width; ++j)
        {
            const auto pixel_index = i * dataset.image_width + j;
            auto &[input, output] = dataset.training_pairs[pixel_index];
            const auto x = static_cast<float>(j) /
                           static_cast<float>(dataset.image_width - 1);
            const auto y = static_cast<float>(i) /
                           static_cast<float>(dataset.image_height - 1);
#if 1
            get_fourier_features_positional_encoding(
                input,
                std::max(dataset.image_width, dataset.image_height),
                {x, y});
#else
            input << (two_pi * frequencies * Eigen::Vector2f {x, y})
                         .array()
                         .cos(),
                (two_pi * frequencies * Eigen::Vector2f {x, y}).array().sin();
#endif
            output = {u8_to_float(image_data[pixel_index * 3 + 0]),
                      u8_to_float(image_data[pixel_index * 3 + 1]),
                      u8_to_float(image_data[pixel_index * 3 + 2])};
        }
    }

    return dataset;
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

[[nodiscard]] SDL_Rect get_target_rect(SDL_Renderer *renderer,
                                       std::size_t image_width,
                                       std::size_t image_height)
{
    SDL_Rect viewport {};
    SDL_RenderGetViewport(renderer, &viewport);

    const auto image_aspect_ratio =
        static_cast<float>(image_width) / static_cast<float>(image_height);
    const auto target_aspect_ratio =
        static_cast<float>(viewport.w) / static_cast<float>(viewport.h);

    auto rect = viewport;
    if (target_aspect_ratio >= image_aspect_ratio)
    {
        rect.w =
            static_cast<int>(static_cast<float>(rect.h) * image_aspect_ratio);
        rect.x = (viewport.w - rect.w) / 2;
    }
    else
    {
        rect.h =
            static_cast<int>(static_cast<float>(rect.w) / image_aspect_ratio);
        rect.y = (viewport.h - rect.h) / 2;
    }

    return rect;
}

void application_main(Network &network,
                      const Training_dataset &dataset,
                      Network::Input &input,
                      float learning_rate)
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

    // TODO: decouple learning speed from framerate, and re-enable V-sync
    const auto renderer =
        SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    sdl_check(renderer);
    DEFER([renderer] { SDL_DestroyRenderer(renderer); });

    const auto texture =
        SDL_CreateTexture(renderer,
                          SDL_PIXELFORMAT_ABGR8888,
                          SDL_TEXTUREACCESS_STREAMING,
                          static_cast<int>(dataset.image_width),
                          static_cast<int>(dataset.image_height));
    sdl_check(texture);
    DEFER([texture] { SDL_DestroyTexture(texture); });

    for (;;)
    {
        SDL_Event e;
        bool quit {false};
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
            {
                quit = true;
            }
        }
        if (quit)
        {
            break;
        }

        Uint8 *pixels;
        int pitch;
        sdl_check(SDL_LockTexture(
            texture, nullptr, reinterpret_cast<void **>(&pixels), &pitch));

        for (std::size_t i {0}; i < dataset.image_height; ++i)
        {
            for (std::size_t j {0}; j < dataset.image_width; ++j)
            {
                const auto pixel_index = i * dataset.image_width + j;
                input = dataset.training_pairs[pixel_index].input;
                network_predict(network, input);
                const auto &prediction = network.output_layer.activations;
                pixels[pixel_index * 4 + 0] = float_to_u8(prediction[0]);
                pixels[pixel_index * 4 + 1] = float_to_u8(prediction[1]);
                pixels[pixel_index * 4 + 2] = float_to_u8(prediction[2]);
                pixels[pixel_index * 4 + 3] = SDL_ALPHA_OPAQUE;
            }
        }

        for (const auto &training_pair : dataset.training_pairs)
        {
            input << training_pair.input;
            network_predict(network, input);
            network_update_weights(
                network, input, training_pair.output, learning_rate);
        }

        SDL_UnlockTexture(texture);

        sdl_check(SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE));
        sdl_check(SDL_RenderClear(renderer));

        const auto dest_rect = get_target_rect(
            renderer, dataset.image_width, dataset.image_height);
        sdl_check(SDL_RenderCopy(renderer, texture, nullptr, &dest_rect));

        SDL_RenderPresent(renderer);
    }
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const auto print_usage = [argv] {
            std::cerr << "Usage: " << argv[0]
                      << " <image> <hidden layers sizes ...>";
        };

        if (argc < 3)
        {
            print_usage();
            return EXIT_FAILURE;
        }

        const auto image_file_name = argv[1];

        std::vector<int> sizes;
        sizes.reserve(1 + static_cast<std::size_t>(argc - 2));
        sizes.push_back(input_size);
        for (int i {2}; i < argc; ++i)
        {
            sizes.push_back(std::stoi(argv[i]));
        }

        const auto image = load_image(image_file_name);

        Network network {};
        network_init(network, sizes);
        Network::Input input(input_size);
        Eigen::internal::set_is_malloc_allowed(false);

        application_main(network, image, input, 0.01f);

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
