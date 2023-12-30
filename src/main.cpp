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

struct Training_pair
{
    Eigen::VectorXf input;
    Eigen::VectorXf output;
};

struct Dataset
{
    std::size_t image_width;
    std::size_t image_height;
    std::vector<Training_pair> training_pairs;
};

[[nodiscard]] constexpr float u8_to_float(std::uint8_t u)
{
    return static_cast<float>(u) / 255.0f;
}

[[nodiscard]] constexpr std::uint8_t float_to_u8(float f)
{
    return static_cast<std::uint8_t>(std::clamp(f, 0.0f, 1.0f) * 255.0f);
}

[[nodiscard]] Eigen::VectorXf get_fourier_features_positional_encoding(
    Eigen::Index input_size, std::size_t max_image_dimension, float x, float y)
{
    Eigen::VectorXf result(input_size);

    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;
    const auto max_frequency = static_cast<float>(max_image_dimension) * 0.5f;
    const Eigen::Index m {input_size / 4};
    for (Eigen::Index j {0}; j < m; ++j)
    {
        const auto frequency = std::pow(
            max_frequency, static_cast<float>(j) / static_cast<float>(m - 1));
        result(4 * j + 0) = std::cos(two_pi * frequency * x);
        result(4 * j + 1) = std::cos(two_pi * frequency * y);
        result(4 * j + 2) = std::sin(two_pi * frequency * x);
        result(4 * j + 3) = std::sin(two_pi * frequency * y);
    }
    return result;
}

[[nodiscard]] Dataset load_dataset(const char *file_name,
                                   Eigen::Index input_size)
{
    int width {};
    int height {};
    int channels_in_file {};
    constexpr int desired_channels {3};

    struct image_deleter
    {
        void operator()(stbi_uc *pointer)
        {
            stbi_image_free(pointer);
        }
    };
    std::unique_ptr<stbi_uc[], image_deleter> image(stbi_load(
        file_name, &width, &height, &channels_in_file, desired_channels));
    if (!image)
    {
        throw std::runtime_error(stbi_failure_reason());
    }

    Dataset dataset {.image_width = static_cast<std::size_t>(width),
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
    Eigen::MatrixX2f frequencies(input_size / 2, 2);
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
            input = get_fourier_features_positional_encoding(
                input_size,
                std::max(dataset.image_width, dataset.image_height),
                x,
                y);
#else
            input << (two_pi * frequencies * Eigen::Vector2f {x, y})
                         .array()
                         .cos(),
                (two_pi * frequencies * Eigen::Vector2f {x, y}).array().sin();
#endif
            output.resize(3);
            output << u8_to_float(image[pixel_index * 3 + 0]),
                u8_to_float(image[pixel_index * 3 + 1]),
                u8_to_float(image[pixel_index * 3 + 2]);
        }
    }

    return dataset;
}

void store_image(std::vector<Layer> &layers,
                 const Dataset &dataset,
                 const char *file_name)
{
    std::cout << "Storing image \"" << file_name << "\"\n";

    std::vector<std::uint8_t> pixel_data(dataset.image_width *
                                         dataset.image_height * 4);

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
            const auto input = get_fourier_features_positional_encoding(
                layers.front().weights.cols(),
                std::max(dataset.image_width, dataset.image_height),
                x,
                y);

            network_predict(layers, input);

            const auto pixel_index = i * dataset.image_width + j;
            pixel_data[pixel_index * 4 + 0] =
                float_to_u8(layers.back().activations(0));
            pixel_data[pixel_index * 4 + 1] =
                float_to_u8(layers.back().activations(1));
            pixel_data[pixel_index * 4 + 2] =
                float_to_u8(layers.back().activations(2));
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

        std::string input_file_name {};
        std::string output_file_name {"out.png"};
        unsigned int num_epochs {1};
        std::vector<unsigned int> layer_sizes {128, 128, 128};
        float learning_rate {0.01f};

        app.add_option("input", input_file_name, "The input image")
            ->required()
            ->check(CLI::ExistingFile);
        app.add_option("-o,--output", output_file_name, "The output image")
            ->capture_default_str();
        app.add_option("-e,--epochs", num_epochs, "Number of training epochs")
            ->capture_default_str();
        app.add_option("-a,--arch", layer_sizes, "Sizes of the network layers")
            ->capture_default_str();
        app.add_option("-l,--learning-rate", learning_rate, "Learning rate")
            ->capture_default_str();

        CLI11_PARSE(app, argc, argv)

        std::cout << "Input: \"" << input_file_name << "\"\n"
                  << "Output: \"" << output_file_name << "\"\n"
                  << "Epochs: " << num_epochs << '\n'
                  << "Network layout: ";
        for (const auto size : layer_sizes)
        {
            std::cout << size << ' ';
        }
        std::cout << "3\n";

        const auto dataset =
            load_dataset(input_file_name.c_str(), layer_sizes.front());

        std::vector<Layer> layers;
        network_init(layers, layer_sizes);

        Eigen::internal::set_is_malloc_allowed(false);

        for (unsigned int epoch {0}; epoch < num_epochs; ++epoch)
        {
            std::cout << "Epoch " << epoch << '\n';
            for (const auto &training_pair : dataset.training_pairs)
            {
                network_predict(layers, training_pair.input);
                network_update_weights(layers,
                                       training_pair.input,
                                       training_pair.output,
                                       learning_rate);
            }
        }

        // FIXME: ideally there should be no allocation from Eigen in
        // store_image()
        Eigen::internal::set_is_malloc_allowed(true);

        store_image(layers, dataset, output_file_name.c_str());

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
