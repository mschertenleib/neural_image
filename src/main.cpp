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
#include <optional>
#include <random>
#include <vector>

#include <cstdlib>

namespace
{

struct Parameters
{
    std::string input_file_name;
    std::string output_file_name;
    std::vector<int> layer_sizes;
    bool force_gray;
    int num_epochs;
    int batch_size;
    float learning_rate;
    int output_width;
    int output_height;
};

struct Dataset
{
    Eigen::MatrixXf inputs;
    Eigen::MatrixXf outputs;
    Eigen::MatrixX2f frequencies;
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

[[nodiscard]] Eigen::MatrixX2f generate_gaussian_frequencies(
    Eigen::Index size, int image_width, int image_height, std::minstd_rand &rng)
{
    assert(size % 2 == 0);

    constexpr float std_dev_scale {0.02f};
    const auto std_dev_x = static_cast<float>(image_width) * std_dev_scale;
    const auto std_dev_y = static_cast<float>(image_height) * std_dev_scale;
    std::normal_distribution<float> distribution_x(0.0f, std_dev_x);
    std::normal_distribution<float> distribution_y(0.0f, std_dev_y);

    Eigen::MatrixX2f frequencies(size / 2, 2);
    frequencies.col(0) = Eigen::VectorXf::NullaryExpr(
        size / 2, [&] { return distribution_x(rng); });
    frequencies.col(1) = Eigen::VectorXf::NullaryExpr(
        size / 2, [&] { return distribution_y(rng); });

    return frequencies;
}

void get_fourier_features(Eigen::MatrixXf &result,
                          const Eigen::MatrixX2f &frequencies,
                          const Eigen::Matrix2Xf &coords)
{
    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;

    result.topRows(result.rows() / 2).noalias() = frequencies * coords;
    result.bottomRows(result.rows() / 2) = result.topRows(result.rows() / 2);
    result.topRows(result.rows() / 2) =
        (two_pi * result.topRows(result.rows() / 2)).array().cos();
    result.bottomRows(result.rows() / 2) =
        (two_pi * result.bottomRows(result.rows() / 2)).array().sin();
}

[[nodiscard]] Dataset load_dataset(const char *file_name,
                                   int input_size,
                                   bool force_gray,
                                   std::minstd_rand &rng)
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

    Dataset dataset {.inputs = {},
                     .outputs = {},
                     .frequencies = generate_gaussian_frequencies(
                         input_size, width, height, rng),
                     .width = width,
                     .height = height,
                     .channels = desired_channels};
    dataset.inputs.setZero(input_size, width * height);
    dataset.outputs.setZero(desired_channels, width * height);

    Eigen::MatrixXf input(input_size, 1);
    Eigen::Matrix2Xf coords(2, 1);
    Eigen::VectorXf output(desired_channels);

#ifndef NDEBUG
    Eigen::internal::set_is_malloc_allowed(false);
#endif

    for (int i {0}; i < height; ++i)
    {
        for (int j {0}; j < width; ++j)
        {
            const auto pixel_index = i * width + j;
            const auto x =
                static_cast<float>(j) / static_cast<float>(width - 1);
            const auto y =
                static_cast<float>(i) / static_cast<float>(height - 1);

            coords << x, y;
            get_fourier_features(input, dataset.frequencies, coords);
            dataset.inputs.col(pixel_index) = input;

            for (int channel {0}; channel < desired_channels; ++channel)
            {
                const auto index =
                    static_cast<std::size_t>(pixel_index) *
                        static_cast<std::size_t>(desired_channels) +
                    static_cast<std::size_t>(channel);
                dataset.outputs(channel, pixel_index) =
                    u8_to_float(image[index]);
            }
        }
    }

#ifndef NDEBUG
    Eigen::internal::set_is_malloc_allowed(true);
#endif

    return dataset;
}

void store_image(const char *file_name,
                 std::vector<Layer> &layers,
                 const Eigen::MatrixX2f &frequencies,
                 int width,
                 int height)
{
    std::cout << "Creating " << width << " x " << height << " output\n";

    const auto input_size = layers.front().weights.cols();
    // FIXME
    const auto batch_size = 1; // layers.front().activations.cols();
    Eigen::MatrixXf input(input_size, batch_size);
    Eigen::Matrix2Xf coords(2, batch_size);

    const auto channels = static_cast<int>(layers.back().activations.size());
    std::vector<std::uint8_t> pixel_data(static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(height) *
                                         static_cast<std::size_t>(channels));

#ifndef NDEBUG
    Eigen::internal::set_is_malloc_allowed(false);
#endif

    for (int i {0}; i < height; ++i)
    {
        for (int j {0}; j < width; ++j)
        {
            const auto x =
                static_cast<float>(j) / static_cast<float>(width - 1);
            const auto y =
                static_cast<float>(i) / static_cast<float>(height - 1);

            coords << x, y;
            get_fourier_features(input, frequencies, coords);

            forward_pass(layers, input);

            const auto pixel_index = i * width + j;
            const auto &output = layers.back().activations.col(0);
            for (int channel {0}; channel < channels; ++channel)
            {
                const auto index = static_cast<std::size_t>(pixel_index) *
                                       static_cast<std::size_t>(channels) +
                                   static_cast<std::size_t>(channel);
                pixel_data[index] = float_to_u8(output(channel));
            }
        }
    }

#ifndef NDEBUG
    Eigen::internal::set_is_malloc_allowed(true);
#endif

    std::cout << "Saving output to " << std::quoted(file_name) << '\n';

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

[[nodiscard]] Parameters parse_command_line(int argc, char *argv[])
{
    Parameters params {.input_file_name = {},
                       .output_file_name = {},
                       .layer_sizes = {},
                       .force_gray = false,
                       .num_epochs = 1,
                       .batch_size = 16,
                       .learning_rate = 0.002f,
                       .output_width = 0,
                       .output_height = 0};

    bool show_help {false};
    std::vector<std::string> unmatched;

    const auto cli =
        (clipp::option("-h", "--help")
             .set(show_help)
             .doc("Show this message and exit") |
         ((clipp::required("-i", "--input") &
           clipp::value(
               clipp::match::prefix_not("-"), "input", params.input_file_name))
              .doc("The input image (JPEG, PNG, TGA, BMP, PSD, GIF, HDR, "
                   "PIC, PNM)"),
          (clipp::required("-o", "--output") &
           clipp::value(clipp::match::prefix_not("-"),
                        "output",
                        params.output_file_name))
              .doc("The output image (PNG)"),
          (clipp::required("-a", "--arch") &
           clipp::values(
               clipp::match::integers(), "layer_sizes", params.layer_sizes))
              .doc("Sizes of the network layers (includes the input "
                   "size but excludes the output size)"),
          (clipp::option("-W", "--width") &
           clipp::value(clipp::match::integers(), "width", params.output_width))
              .doc("The width of the output image (if neither width nor "
                   "height are specified, defaults to the dimension of the "
                   "input image. If only one of width or height is specified, "
                   "keeps the same aspect ratio as the input image)"),
          (clipp::option("-H", "--height") &
           clipp::value(
               clipp::match::integers(), "height", params.output_height))
              .doc("The height of the output image (if neither width nor "
                   "height are specified, defaults to the dimension of the "
                   "input image. If only one of width or height is specified, "
                   "keeps the same aspect ratio as the input image)"),
          clipp::option("-g", "--gray")
              .set(params.force_gray)
              .doc("Force grayscale (by default, the output will be either "
                   "RGB or grayscale depending on the input)"),
          (clipp::option("-e", "--epochs") &
           clipp::value(clipp::match::integers(), "epochs", params.num_epochs))
              .doc("Number of training epochs (default: " +
                   std::to_string(params.num_epochs) + ")"),
          (clipp::option("-b", "--batch-size") &
           clipp::value(
               clipp::match::integers(), "batch_size", params.batch_size))
              .doc("Mini-batch size (default: " +
                   std::to_string(params.batch_size) + ")"),
          (clipp::option("-l", "--learning-rate") &
           clipp::value(
               clipp::match::numbers(), "learning_rate", params.learning_rate))
              .doc("Learning rate (default: " +
                   std::to_string(params.learning_rate) + ")"),
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
        std::exit(EXIT_FAILURE);
    }

    if (show_help)
    {
        std::cout << clipp::make_man_page(
                         cli,
                         std::filesystem::path(argv[0]).filename().string())
                  << '\n';
        std::exit(EXIT_SUCCESS);
    }

    for (const auto size : params.layer_sizes)
    {
        if (size <= 0)
        {
            std::cerr << "Error on layer size of " << size
                      << ": must be strictly positive\n";
            std::exit(EXIT_FAILURE);
        }
    }
    if (params.layer_sizes.front() % 2 != 0)
    {
        std::cerr << "Error on network input size of "
                  << params.layer_sizes.front() << ": must be divisible by 2\n";
        std::exit(EXIT_FAILURE);
    }
    if (params.num_epochs < 0)
    {
        std::cerr << "Error on number of epochs of " << params.num_epochs
                  << ": must be positive\n";
        std::exit(EXIT_FAILURE);
    }
    if (params.batch_size <= 0)
    {
        std::cerr << "Error on mini-batch size of " << params.batch_size
                  << ": must be strictly positive\n";
        std::exit(EXIT_FAILURE);
    }
    if (params.output_width < 0)
    {
        std::cerr << "Error on output width of " << params.output_width
                  << ": must be positive\n";
        std::exit(EXIT_FAILURE);
    }
    if (params.output_height < 0)
    {
        std::cerr << "Error on output height of " << params.output_height
                  << ": must be positive\n";
        std::exit(EXIT_FAILURE);
    }

    return params;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        auto params = parse_command_line(argc, argv);

        std::cout << "Input: " << std::quoted(params.input_file_name) << '\n'
                  << "Output: " << std::quoted(params.output_file_name) << '\n'
                  << "Network layout:";
        for (const auto size : params.layer_sizes)
        {
            std::cout << ' ' << size;
        }
        std::cout << '\n'
                  << "Epochs: " << params.num_epochs << '\n'
                  << "Mini-batch size: " << params.batch_size << '\n'
                  << "Learning rate: " << params.learning_rate << '\n';

        std::random_device rd;
        std::minstd_rand rng(rd());

        std::cout.flush();
        std::cerr << "TODO: mini-batch with size " << params.batch_size
                  << " not implemented" << std::endl;
        params.batch_size = 1;

        const auto dataset = load_dataset(params.input_file_name.c_str(),
                                          params.layer_sizes.front(),
                                          params.force_gray,
                                          rng);
        params.layer_sizes.push_back(dataset.channels);

        if (params.output_width > 0 && params.output_height <= 0)
        {
            const auto aspect_ratio = static_cast<float>(dataset.width) /
                                      static_cast<float>(dataset.height);
            params.output_height = static_cast<int>(
                static_cast<float>(params.output_width) / aspect_ratio);
            params.output_height = std::max(params.output_height, 1);
        }
        else if (params.output_width <= 0 && params.output_height > 0)
        {
            const auto aspect_ratio = static_cast<float>(dataset.width) /
                                      static_cast<float>(dataset.height);
            params.output_width = static_cast<int>(
                static_cast<float>(params.output_height) * aspect_ratio);
            params.output_width = std::max(params.output_width, 1);
        }
        else if (params.output_width <= 0 && params.output_height <= 0)
        {
            params.output_width = dataset.width;
            params.output_height = dataset.height;
        }

        std::cout << "Channels: " << dataset.channels
                  << (dataset.channels < 3 ? " (grayscale)\n" : " (RGB)\n");
        std::cout << "Input is " << dataset.width << " x " << dataset.height
                  << " pixels\n"
                  << "Output is " << params.output_width << " x "
                  << params.output_height << " pixels\n"
                  << std::string(72, '-') << '\n';

        std::vector<int> indices(
            static_cast<std::size_t>(dataset.inputs.cols()));
        std::iota(indices.begin(), indices.end(), 0);

        auto layers = network_init(params.layer_sizes, params.batch_size, rng);

        Eigen::MatrixXf input(params.layer_sizes.front(), params.batch_size);
        Eigen::MatrixXf output(params.layer_sizes.back(), params.batch_size);

#ifndef NDEBUG
        Eigen::internal::set_is_malloc_allowed(false);
#endif

        for (int epoch {0}; epoch < params.num_epochs; ++epoch)
        {
            std::cout << "Training epoch " << epoch << '\n';

            std::shuffle(indices.begin(), indices.end(), rng);
            for (const auto index : indices)
            {
                input << dataset.inputs.col(index);
                output << dataset.outputs.col(index);
                training_pass(layers,
                              input,
                              output,
                              params.batch_size,
                              params.learning_rate);
            }
        }

#ifndef NDEBUG
        Eigen::internal::set_is_malloc_allowed(true);
#endif

        store_image(params.output_file_name.c_str(),
                    layers,
                    dataset.frequencies,
                    params.output_width,
                    params.output_height);

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
