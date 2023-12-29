#include "network.hpp"

#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "CLI/CLI.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <vector>

#include <cstdlib>

namespace
{

constexpr int input_size {128};

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

struct Free_image
{
    void operator()(stbi_uc *pointer) const
    {
        stbi_image_free(pointer);
    }
};

[[nodiscard]] constexpr float u8_to_float(std::uint8_t u)
{
    return static_cast<float>(u) / 255.0f;
}

[[nodiscard]] constexpr std::uint8_t float_to_u8(float f)
{
    return static_cast<std::uint8_t>(std::clamp(f, 0.0f, 1.0f) * 255.0f);
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

[[nodiscard]] Training_dataset load_dataset(const char *file_name)
{
    std::cout << "Loading image \"" << file_name << "\"\n";

    int width {};
    int height {};
    int channels_in_file {};
    constexpr int desired_channels {3};
    std::unique_ptr<stbi_uc[], Free_image> image {stbi_load(
        file_name, &width, &height, &channels_in_file, desired_channels)};
    if (!image)
    {
        throw std::runtime_error(stbi_failure_reason());
    }

    Training_dataset dataset {.image_width = static_cast<std::size_t>(width),
                              .image_height = static_cast<std::size_t>(height),
                              .training_pairs = {}};
    dataset.training_pairs.resize(dataset.image_width * dataset.image_height);

    std::random_device rd {};
    std::minstd_rand rng(rd());
    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;
    const auto std_dev = static_cast<float>(std::max(dataset.image_width,
                                                     dataset.image_height)) *
                         0.01f;
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
            output = {u8_to_float(image[pixel_index * 3 + 0]),
                      u8_to_float(image[pixel_index * 3 + 1]),
                      u8_to_float(image[pixel_index * 3 + 2])};
        }
    }

    return dataset;
}

void store_image(Network &network,
                 const Training_dataset &dataset,
                 const char *file_name)
{
    std::cout << "Storing image \"" << file_name << "\"\n";

    std::vector<std::uint8_t> pixel_data(dataset.image_width *
                                         dataset.image_height * 4);

    Eigen::Vector<float, input_size> input;

    for (std::size_t i {0}; i < dataset.image_height; ++i)
    {
        for (std::size_t j {0}; j < dataset.image_width; ++j)
        {
            const auto x = static_cast<float>(j) /
                           static_cast<float>(dataset.image_width - 1);
            const auto y = static_cast<float>(i) /
                           static_cast<float>(dataset.image_height - 1);

            // FIXME: these have to be the same features (frequencies) used for
            // training, we really should store them somewhere and not call this
            // function again
            get_fourier_features_positional_encoding(
                input,
                std::max(dataset.image_width, dataset.image_height),
                {x, y});

            network_predict(network, input);

            const auto pixel_index = i * dataset.image_width + j;
            pixel_data[pixel_index * 4 + 0] =
                float_to_u8(network.output_layer.activations(0));
            pixel_data[pixel_index * 4 + 1] =
                float_to_u8(network.output_layer.activations(1));
            pixel_data[pixel_index * 4 + 2] =
                float_to_u8(network.output_layer.activations(2));
            pixel_data[pixel_index * 4 + 3] = 255;
        }
    }

    const auto write_result =
        stbi_write_png(file_name,
                       static_cast<int>(dataset.image_width),
                       static_cast<int>(dataset.image_height),
                       4,
                       pixel_data.data(),
                       static_cast<int>(dataset.image_width) * 4);
    if (write_result == 0)
    {
        throw std::runtime_error("Failed to store image");
    }
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        CLI::App app;
        argv = app.ensure_utf8(argv);

        std::string input_file_name;
        std::string output_file_name;
        int num_epochs {1};

        app.add_option("-i,--input", input_file_name, "The input image");
        app.add_option("-o,--output", output_file_name, "The output image");
        app.add_option("--epochs", num_epochs, "Number of training epochs");

        CLI11_PARSE(app, argc, argv)

        // FIXME
        const std::vector<int> sizes {input_size, 128, 128};

        std::cout << "Input: \"" << input_file_name << "\"\n"
                  << "Output: \"" << output_file_name << "\"\n"
                  << "Epochs: " << num_epochs << '\n'
                  << "Network layout: ";
        for (const auto size : sizes)
        {
            std::cout << size << ' ';
        }
        std::cout << "3\n";

        const auto dataset = load_dataset(input_file_name.c_str());

        Network network {};
        network_init(network, sizes);
        Network::Input input(input_size);
        Eigen::internal::set_is_malloc_allowed(false);

        constexpr float learning_rate {0.01f};

        for (int epoch {0}; epoch < num_epochs; ++epoch)
        {
            std::cout << "Epoch " << epoch << '\n';
            for (const auto &training_pair : dataset.training_pairs)
            {
                input << training_pair.input;
                network_predict(network, input);
                network_update_weights(
                    network, input, training_pair.output, learning_rate);
            }
        }

        store_image(network, dataset, output_file_name.c_str());

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
