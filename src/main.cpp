#include "network.hpp"

#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "CLI/CLI.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <vector>

#include <cstdlib>

namespace
{

class Custom_formatter : public CLI::Formatter
{
public:
    std::string make_option_opts(const CLI::Option *opt) const override
    {
        std::stringstream out;

        if (!opt->get_option_text().empty())
        {
            out << " " << opt->get_option_text();
        }
        else
        {
            if (opt->get_type_size() != 0)
            {
                if (!opt->get_type_name().empty())
                    out << " " << get_label(opt->get_type_name());
                if (!opt->get_default_str().empty())
                    out << " [" << opt->get_default_str() << "] ";
                if (opt->get_expected_max() ==
                    CLI::detail::expected_max_vector_size)
                    out << " ...";
                else if (opt->get_expected_min() > 1)
                    out << " x " << opt->get_expected();

                if (opt->get_required())
                    out << " " << get_label("REQUIRED");
            }
            if (!opt->get_envname().empty())
                out << " (" << get_label("Env") << ":" << opt->get_envname()
                    << ")";
            if (!opt->get_needs().empty())
            {
                out << " " << get_label("Needs") << ":";
                for (const CLI::Option *op : opt->get_needs())
                    out << " " << op->get_name();
            }
            if (!opt->get_excludes().empty())
            {
                out << " " << get_label("Excludes") << ":";
                for (const CLI::Option *op : opt->get_excludes())
                    out << " " << op->get_name();
            }
        }
        return out.str();
    }
};

struct Dataset
{
    Eigen::MatrixXf inputs;
    Eigen::MatrixXf outputs;
    int width;
    int height;
};

[[nodiscard]] constexpr float u8_to_float(std::uint8_t u) noexcept
{
    return static_cast<float>(u) / 255.0f;
}

[[nodiscard]] constexpr std::uint8_t float_to_u8(float f) noexcept
{
    return static_cast<std::uint8_t>(std::clamp(f, 0.0f, 1.0f) * 255.0f);
}

void get_fourier_features_positional_encoding(Eigen::VectorXf &result,
                                              Eigen::Index max_image_dimension,
                                              float x,
                                              float y)
{
    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;
    const auto max_frequency = static_cast<float>(max_image_dimension) * 0.5f;
    const Eigen::Index m {result.size() / 4};
    for (Eigen::Index j {0}; j < m; ++j)
    {
        const auto frequency = std::pow(
            max_frequency, static_cast<float>(j) / static_cast<float>(m - 1));
        result(4 * j + 0) = std::cos(two_pi * frequency * x);
        result(4 * j + 1) = std::cos(two_pi * frequency * y);
        result(4 * j + 2) = std::sin(two_pi * frequency * x);
        result(4 * j + 3) = std::sin(two_pi * frequency * y);
    }
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

    Dataset dataset {
        .inputs = {}, .outputs = {}, .width = width, .height = height};
    dataset.inputs.setZero(input_size, width * height);
    constexpr Eigen::Index output_size {3};
    dataset.outputs.setZero(output_size, width * height);

    std::random_device rd {};
    std::minstd_rand rng(rd());
    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;
    const auto std_dev = static_cast<float>(std::max(width, height)) * 0.01f;
    std::normal_distribution<float> distribution(0.0f, std_dev);
    const auto generate_weight = [&](float) { return distribution(rng); };
    Eigen::MatrixX2f frequencies(input_size / 2, 2);
    frequencies = frequencies.unaryExpr(generate_weight);

    Eigen::VectorXf input(input_size);
    Eigen::VectorXf output(output_size);

    for (Eigen::Index i {0}; i < height; ++i)
    {
        for (Eigen::Index j {0}; j < width; ++j)
        {
            const auto pixel_index = i * width + j;
            const auto x =
                static_cast<float>(j) / static_cast<float>(width - 1);
            const auto y =
                static_cast<float>(i) / static_cast<float>(height - 1);
#if 1
            get_fourier_features_positional_encoding(
                input, std::max(width, height), x, y);
#else
            input << (two_pi * frequencies * Eigen::Vector2f {x, y})
                         .array()
                         .cos(),
                (two_pi * frequencies * Eigen::Vector2f {x, y}).array().sin();
#endif
            output << u8_to_float(
                image[static_cast<std::size_t>(pixel_index * 3 + 0)]),
                u8_to_float(
                    image[static_cast<std::size_t>(pixel_index * 3 + 1)]),
                u8_to_float(
                    image[static_cast<std::size_t>(pixel_index * 3 + 2)]);

            dataset.inputs.col(pixel_index) = input;
            dataset.outputs.col(pixel_index) = output;
        }
    }

    return dataset;
}

void store_image(std::vector<Layer> &layers,
                 const Dataset &dataset,
                 std::size_t width,
                 std::size_t height,
                 const char *file_name)
{
    std::vector<std::uint8_t> pixel_data(width * height * 4);

    const auto input_size = layers.front().weights.cols();
    Eigen::VectorXf input(input_size);

    for (std::size_t i {0}; i < height; ++i)
    {
        for (std::size_t j {0}; j < width; ++j)
        {
            // FIXME: x and y should be in the range [-1, 1]
            const auto x =
                static_cast<float>(j) / static_cast<float>(width - 1);
            const auto y =
                static_cast<float>(i) / static_cast<float>(height - 1);

            get_fourier_features_positional_encoding(
                input, std::max(width, height), x, y);

            forward_pass(layers, input);

            const auto index = i * width + j;
            const auto &output = layers.back().activations;
            pixel_data[index * 4 + 0] = float_to_u8(output(0));
            pixel_data[index * 4 + 1] = float_to_u8(output(1));
            pixel_data[index * 4 + 2] = float_to_u8(output(2));
            pixel_data[index * 4 + 3] = 255;
        }
    }

    const auto write_result = stbi_write_png(file_name,
                                             static_cast<int>(width),
                                             static_cast<int>(height),
                                             4,
                                             pixel_data.data(),
                                             static_cast<int>(width) * 4);
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
        CLI::App cli_app;
        argv = cli_app.ensure_utf8(argv);

        std::string input_file_name {};
        std::string output_file_name {"out.png"};
        unsigned int num_epochs {1};
        std::vector<Eigen::Index> layer_sizes {128, 128, 128};
        float learning_rate {0.01f};

        cli_app.add_option("input", input_file_name, "The input image")
            ->required()
            ->check(CLI::ExistingFile);
        cli_app
            .add_option(
                "-o,--output", output_file_name, "The output image (PNG)")
            ->capture_default_str();
        cli_app
            .add_option("-e,--epochs", num_epochs, "Number of training epochs")
            ->capture_default_str();
        cli_app
            .add_option("-a,--arch",
                        layer_sizes,
                        "Sizes of the network layers (includes the input size "
                        "but excludes the output size)")
            ->capture_default_str()
            ->check(CLI::Range(Eigen::Index {1},
                               std::numeric_limits<Eigen::Index>::max()))
            ->check(CLI::Range(Eigen::Index {2},
                               std::numeric_limits<Eigen::Index>::max())
                        .application_index(0));
        cli_app
            .add_option("-l,--learning-rate", learning_rate, "Learning rate")
            ->capture_default_str();

        cli_app.formatter(std::make_shared<Custom_formatter>());

        CLI11_PARSE(cli_app, argc, argv)

        // TODO: we should let the user select 1 or 3 output channels
        layer_sizes.push_back(3);

        std::cout << "Input: \"" << input_file_name << "\"\n"
                  << "Output: \"" << output_file_name << "\"\n"
                  << "Epochs: " << num_epochs << '\n'
                  << "Network layout:";
        for (const auto size : layer_sizes)
        {
            std::cout << ' ' << size;
        }
        std::cout << '\n';

        const auto dataset =
            load_dataset(input_file_name.c_str(), layer_sizes.front());

        std::random_device rd;
        std::minstd_rand rng(rd());
        std::vector<Eigen::Index> indices(
            static_cast<std::size_t>(dataset.width) *
            static_cast<std::size_t>(dataset.height));
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<Layer> layers;
        network_init(layers, layer_sizes);

        Eigen::VectorXf input(layer_sizes.front());
        Eigen::VectorXf output(layer_sizes.back());

#ifndef NDEBUG
        Eigen::internal::set_is_malloc_allowed(false);
#endif

        for (unsigned int epoch {0}; epoch < num_epochs; ++epoch)
        {
            std::cout << "Epoch " << epoch << '\n';

            std::shuffle(indices.begin(), indices.end(), rng);
            for (const auto index : indices)
            {
                input << dataset.inputs.col(index);
                output << dataset.outputs.col(index);
                forward_pass(layers, input);
                backward_pass(layers, input, output, learning_rate);
            }
        }

#ifndef NDEBUG
        Eigen::internal::set_is_malloc_allowed(true);
#endif

        store_image(layers,
                    dataset,
                    static_cast<std::size_t>(dataset.width),
                    static_cast<std::size_t>(dataset.height),
                    output_file_name.c_str());

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
