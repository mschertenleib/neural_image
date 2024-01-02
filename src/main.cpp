#include "network.hpp"

#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// NOTE: clipp uses std::result_of, but it is removed in C++20. GCC did not
// remove it yet, so just define it for MSVC.
#ifdef _MSC_VER
namespace std
{
template <class>
struct result_of;
template <class F, class... ArgTypes>
struct result_of<F(ArgTypes...)> : std::invoke_result<F, ArgTypes...>
{
};
} // namespace std
#endif
#include "clipp.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numbers>
#include <numeric>
#include <random>
#include <vector>

#include <cstdlib>

namespace
{

struct Dataset
{
    Eigen::MatrixXf inputs;
    Eigen::MatrixXf outputs;
    int width;
    int height;
    int channels;
};

[[nodiscard]] constexpr float u8_to_float(std::uint8_t u) noexcept
{
    return static_cast<float>(u) / 255.0f;
}

[[nodiscard]] constexpr std::uint8_t float_to_u8(float f) noexcept
{
    return static_cast<std::uint8_t>(std::clamp(f, 0.0f, 1.0f) * 255.0f);
}

[[nodiscard]] Eigen::MatrixX2f generate_gaussian_frequencies(Eigen::Index size,
                                                             int image_width,
                                                             int image_height)
{
    assert(size % 2 == 0);

    std::random_device rd;
    std::minstd_rand rng(rd());

    const auto std_dev_x = static_cast<float>(image_width) * 0.02f;
    const auto std_dev_y = static_cast<float>(image_height) * 0.02f;
    std::normal_distribution<float> distribution_x(0.0f, std_dev_x);
    std::normal_distribution<float> distribution_y(0.0f, std_dev_y);

    Eigen::MatrixX2f frequencies(size / 2, 2);
    frequencies.col(0) = Eigen::VectorXf::NullaryExpr(
        size / 2, [&] { return distribution_x(rng); });
    frequencies.col(1) = Eigen::VectorXf::NullaryExpr(
        size / 2, [&] { return distribution_y(rng); });

    return frequencies;
}

void get_fourier_features(Eigen::VectorXf &result,
                          int image_width,
                          int image_height,
                          float x,
                          float y)
{
#if 0

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

#else

    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;

    // FIXME: call this outside of the function to remove the need for a static
    // variable
    static const auto frequencies =
        generate_gaussian_frequencies(result.size(), image_width, image_height);

    result << (two_pi * frequencies * Eigen::Vector2f {x, y}).array().cos(),
        (two_pi * frequencies * Eigen::Vector2f {x, y}).array().sin();

#endif
}

[[nodiscard]] Dataset
load_dataset(const char *file_name, Eigen::Index input_size, bool force_gray)
{
    int width {};
    int height {};
    int channels_in_file {};
    if (!stbi_info(file_name, &width, &height, &channels_in_file))
    {
        throw std::runtime_error(stbi_failure_reason());
    }

    int desired_channels {3};
    if (channels_in_file < 3 || force_gray)
    {
        desired_channels = 1;
    }

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

    std::cout << "Input size is " << width << " x " << height << " pixels, "
              << desired_channels << " channels\n";

    Dataset dataset {.inputs = {},
                     .outputs = {},
                     .width = width,
                     .height = height,
                     .channels = desired_channels};
    dataset.inputs.setZero(input_size, width * height);
    dataset.outputs.setZero(desired_channels, width * height);

    Eigen::VectorXf input(input_size);
    Eigen::VectorXf output(desired_channels);

    for (Eigen::Index i {0}; i < height; ++i)
    {
        for (Eigen::Index j {0}; j < width; ++j)
        {
            const auto pixel_index = i * width + j;
            const auto x =
                static_cast<float>(j) / static_cast<float>(width - 1);
            const auto y =
                static_cast<float>(i) / static_cast<float>(height - 1);

            get_fourier_features(input, width, height, x, y);

            for (int channel {0}; channel < desired_channels; ++channel)
            {
                output(channel) = u8_to_float(image[static_cast<std::size_t>(
                    pixel_index * desired_channels + channel)]);
            }
            dataset.inputs.col(pixel_index) = input;
            dataset.outputs.col(pixel_index) = output;
        }
    }

    return dataset;
}

void store_image(const char *file_name,
                 std::vector<Layer> &layers,
                 int width,
                 int height,
                 int input_width,
                 int input_height)
{
    Eigen::VectorXf input(layers.front().weights.cols());

    const auto channels = static_cast<int>(layers.back().activations.size());
    std::vector<std::uint8_t> pixel_data(static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(height) *
                                         static_cast<std::size_t>(channels));

    for (int i {0}; i < height; ++i)
    {
        for (int j {0}; j < width; ++j)
        {
            // FIXME: x and y should be in the range [-1, 1]
            const auto x =
                static_cast<float>(j) / static_cast<float>(width - 1);
            const auto y =
                static_cast<float>(i) / static_cast<float>(height - 1);

            get_fourier_features(input, input_width, input_height, x, y);

            forward_pass(layers, input);

            const auto pixel_index = i * width + j;
            const auto &output = layers.back().activations;
            for (int channel {0}; channel < channels; ++channel)
            {
                pixel_data[static_cast<std::size_t>(pixel_index) *
                               static_cast<std::size_t>(channels) +
                           static_cast<std::size_t>(channel)] =
                    float_to_u8(output(channel));
            }
        }
    }

    const auto write_result = stbi_write_png(file_name,
                                             width,
                                             height,
                                             channels,
                                             pixel_data.data(),
                                             width * channels);
    if (write_result == 0)
    {
        throw std::runtime_error("Failed to store image");
    }
}

void print_error(const clipp::parsing_result &result,
                 const std::vector<std::string> &unmatched,
                 const clipp::group &cli,
                 const std::string &executable_name)
{
    if (!unmatched.empty())
    {
        std::cerr << "Unmatched extra arguments:";
        for (const auto &arg : unmatched)
        {
            std::cerr << " \"" << arg << '\"';
        }
        std::cerr << '\n';
    }

    for (const auto &arg : result.missing())
    {
        if (!arg.param()->label().empty())
        {
            std::cerr << "Missing parameter \"" << arg.param()->label()
                      << "\" after index " << arg.after_index() << '\n';
        }
    }

    for (const auto &arg : result)
    {
        if (arg.any_error())
        {
            std::cerr << "Error at argument " << arg.index() << " \""
                      << arg.arg() << "\"\n";
        }
    }

    std::cerr << "Usage:\n" << clipp::usage_lines(cli, executable_name) << '\n';
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        bool show_help {false};
        std::string input_file_name;
        std::string output_file_name {"out.png"};
        std::vector<int> layer_sizes;
        bool force_gray {false};
        unsigned int num_epochs {1};
        float learning_rate {0.002f};
        int output_width {};
        int output_height {};
        std::vector<std::string> unmatched;

        const auto cli =
            (clipp::option("-h", "--help")
                 .set(show_help)
                 .doc("Show this message and exit") |
             ((clipp::required("-i", "--input") &
               clipp::value(
                   clipp::match::prefix_not("-"), "input", input_file_name))
                  .doc("The input image (JPEG, PNG, TGA, BMP, PSD, GIF, HDR, "
                       "PIC, PNM)"),
              (clipp::required("-o", "--output") &
               clipp::value(
                   clipp::match::prefix_not("-"), "output", output_file_name))
                  .doc("The output image (PNG)"),
              (clipp::option("-W", "--width") &
               clipp::value(
                   clipp::match::positive_integers(), "width", output_width))
                  .doc("The width of the output image"),
              (clipp::option("-H", "--height") &
               clipp::value(
                   clipp::match::positive_integers(), "height", output_height))
                  .doc("The height of the output image"),
              (clipp::option("-a", "--arch") &
               clipp::values(clipp::match::positive_integers(),
                             "layer_sizes",
                             layer_sizes))
                  .doc("Sizes of the network layers (includes the input "
                       "size but excludes the output size)"),
              clipp::option("-g", "--gray")
                  .set(force_gray)
                  .doc("Force grayscale for the output image (by default, the "
                       "output will be either RGB or grayscale depending on "
                       "the input)"),
              (clipp::option("-e", "--epochs") &
               clipp::value(
                   clipp::match::positive_integers(), "epochs", num_epochs))
                  .doc("Number of training epochs"),
              (clipp::option("-l", "--learning_rate") &
               clipp::value(
                   clipp::match::numbers(), "learning_rate", learning_rate))
                  .doc("Learning rate"),
              clipp::any_other(unmatched)));

        assert(cli.flags_are_prefix_free());
        assert(cli.common_flag_prefix() == "-");

        const auto result = clipp::parse(argc, argv, cli);

        if (result.any_error() || !unmatched.empty())
        {
            print_error(result,
                        unmatched,
                        cli,
                        std::filesystem::path(argv[0]).filename().string());
            return EXIT_FAILURE;
        }

        if (show_help)
        {
            std::cout << clipp::make_man_page(
                             cli,
                             std::filesystem::path(argv[0]).filename().string())
                      << '\n';
            return EXIT_SUCCESS;
        }

        if (layer_sizes.empty())
        {
            layer_sizes.assign({128, 64, 64});
        }

        std::cout << "Input: \"" << input_file_name << "\"\n"
                  << "Output: \"" << output_file_name << "\"\n"
                  << "Network layout:";
        for (const auto size : layer_sizes)
        {
            std::cout << ' ' << size;
        }
        std::cout << '\n'
                  << "Epochs: " << num_epochs << '\n'
                  << "Learning rate: " << learning_rate << '\n';

        const auto dataset = load_dataset(
            input_file_name.c_str(), layer_sizes.front(), force_gray);

        layer_sizes.push_back(dataset.channels);

        if (output_width > 0 && output_height == 0)
        {
            const auto aspect_ratio = static_cast<float>(dataset.width) /
                                      static_cast<float>(dataset.height);
            output_height = static_cast<int>(static_cast<float>(output_width) /
                                             aspect_ratio);
        }
        else if (output_width == 0 && output_height > 0)
        {
            const auto aspect_ratio = static_cast<float>(dataset.width) /
                                      static_cast<float>(dataset.height);
            output_width = static_cast<int>(static_cast<float>(output_height) *
                                            aspect_ratio);
        }
        else if (output_width == 0 && output_height == 0)
        {
            output_width = dataset.width;
            output_height = dataset.height;
        }

        std::cout << "Output size is " << output_width << " x " << output_height
                  << " pixels\n";

        std::cout << std::string(72, '-') << '\n';

        std::random_device rd;
        std::minstd_rand rng(rd());
        std::vector<int> indices(
            static_cast<std::size_t>(dataset.inputs.cols()));
        std::iota(indices.begin(), indices.end(), 0);

        auto layers = network_init(layer_sizes);

        Eigen::VectorXf input(layer_sizes.front());
        Eigen::VectorXf output(layer_sizes.back());

        for (unsigned int epoch {0}; epoch < num_epochs; ++epoch)
        {
            std::cout << "Training epoch " << epoch << '\n';

            std::shuffle(indices.begin(), indices.end(), rng);
            for (const auto index : indices)
            {
                input << dataset.inputs.col(index);
                output << dataset.outputs.col(index);
                training_pass(layers, input, output, learning_rate);
            }
        }

        std::cout << "Saving " << output_width << " x " << output_height
                  << " output to " << std::quoted(output_file_name) << '\n';
        store_image(output_file_name.c_str(),
                    layers,
                    output_width,
                    output_height,
                    dataset.width,
                    dataset.height);

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
