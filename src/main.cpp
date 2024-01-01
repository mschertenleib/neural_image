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
#include <ctime>

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

    Dataset dataset {.inputs = {},
                     .outputs = {},
                     .width = width,
                     .height = height,
                     .channels = desired_channels};
    dataset.inputs.setZero(input_size, width * height);
    dataset.outputs.setZero(desired_channels, width * height);

    std::random_device rd {};
    std::minstd_rand rng(rd());
    constexpr auto two_pi = 2.0f * std::numbers::pi_v<float>;
    const auto std_dev = static_cast<float>(std::max(width, height)) * 0.01f;
    std::normal_distribution<float> distribution(0.0f, std_dev);
    const auto generate_weight = [&](float) { return distribution(rng); };
    Eigen::MatrixX2f frequencies(input_size / 2, 2);
    frequencies = frequencies.unaryExpr(generate_weight);

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
#if 1
            get_fourier_features_positional_encoding(
                input, std::max(width, height), x, y);
#else
            input << (two_pi * frequencies * Eigen::Vector2f {x, y})
                         .array()
                         .cos(),
                (two_pi * frequencies * Eigen::Vector2f {x, y}).array().sin();
#endif
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

void store_image(std::vector<Layer> &layers,
                 int width,
                 int height,
                 int channels,
                 const char *file_name)
{
    std::vector<std::uint8_t> pixel_data(static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(height) *
                                         static_cast<std::size_t>(channels));

    const auto input_size = layers.front().weights.cols();
    Eigen::VectorXf input(input_size);

    for (int i {0}; i < height; ++i)
    {
        for (int j {0}; j < width; ++j)
        {
            // FIXME: x and y should be in the range [-1, 1]
            const auto x =
                static_cast<float>(j) / static_cast<float>(width - 1);
            const auto y =
                static_cast<float>(i) / static_cast<float>(height - 1);

            get_fourier_features_positional_encoding(
                input, std::max(width, height), x, y);

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

[[nodiscard]] std::filesystem::path
create_progress_path(const std::string &progress_directory)
{
    const auto t = std::time(nullptr);
    const auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return std::filesystem::path(progress_directory) /
           std::filesystem::path(oss.str());
}

void save_progress(const std::filesystem::path &dir_path,
                   std::vector<Layer> &layers,
                   const Dataset &dataset,
                   unsigned int epoch)
{
    char buffer[] {"00000.png"};
    std::snprintf(buffer, sizeof(buffer), "%05u.png", epoch);
    const auto file_path = dir_path / std::filesystem::path(buffer);
    std::cout << "Saving progress to " << file_path << '\n';
    store_image(layers,
                dataset.width,
                dataset.height,
                dataset.channels,
                file_path.string().c_str());
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        bool show_help {false};
        std::string input_file_name;
        std::string output_file_name {"out.png"};
        std::string progress_directory;
        std::vector<int> layer_sizes;
        bool force_gray {false};
        unsigned int num_epochs {1};
        float learning_rate {0.01f};
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
              (clipp::option("-p", "--progress") &
               clipp::value(clipp::match::prefix_not("-"),
                            "directory",
                            progress_directory))
                  .doc("Save output images at each epoch. From the given "
                       "directory, the images will be stored in a nested "
                       "directory with a timestamp name"),
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
                  << "Output: \"" << output_file_name << "\"\n";
        std::filesystem::path progress_path;
        if (!progress_directory.empty())
        {
            progress_path = create_progress_path(progress_directory);
            std::cout << "Progress directory: " << progress_path << '\n';
            std::filesystem::create_directory(progress_directory);
            std::filesystem::create_directory(progress_path);
        }
        std::cout << "Network layout:";
        for (const auto size : layer_sizes)
        {
            std::cout << ' ' << size;
        }
        std::cout << '\n';
        std::cout << "Epochs: " << num_epochs << '\n';
        std::cout << "Learning rate: " << learning_rate << '\n';

        const auto dataset = load_dataset(
            input_file_name.c_str(), layer_sizes.front(), force_gray);
        layer_sizes.push_back(dataset.channels);

        std::cout << "Channels: " << dataset.channels << '\n';
        std::cout << std::string(72, '-') << '\n';

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
        /*
        #ifndef NDEBUG
                Eigen::internal::set_is_malloc_allowed(false);
        #endif
        */

        for (unsigned int epoch {0}; epoch < num_epochs; ++epoch)
        {
            if (!progress_path.empty())
            {
                save_progress(progress_path, layers, dataset, epoch);
            }

            std::cout << "Training epoch " << epoch << '\n';

            std::shuffle(indices.begin(), indices.end(), rng);
            for (const auto index : indices)
            {
                input << dataset.inputs.col(index);
                output << dataset.outputs.col(index);
                forward_pass(layers, input);
                backward_pass(layers, input, output, learning_rate);
            }
        }
        /*
        #ifndef NDEBUG
                Eigen::internal::set_is_malloc_allowed(true);
        #endif
        */

        if (!progress_path.empty())
        {
            save_progress(progress_path, layers, dataset, num_epochs);
        }

        std::cout << "Saving output to " << std::quoted(output_file_name)
                  << '\n';
        store_image(layers,
                    dataset.width,
                    dataset.height,
                    dataset.channels,
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
