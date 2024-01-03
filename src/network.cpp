#include "network.hpp"

#include <cmath>
#include <random>

namespace
{

inline void
layer_init_zero(Layer &layer, int size, int input_size, int batch_size)
{
    layer.weights.setZero(size, input_size);
    layer.biases.setZero(size);
    layer.activations.setZero(size, batch_size);
    layer.deltas.setZero(size, batch_size);
}

inline void layer_init_leaky_relu(Layer &layer,
                                  int size,
                                  int input_size,
                                  int batch_size,
                                  std::minstd_rand &rng)
{
    layer_init_zero(layer, size, input_size, batch_size);

    const auto std_dev = std::sqrt(2.0f / static_cast<float>(input_size));
    std::normal_distribution<float> distribution(0.0f, std_dev);
    const auto generate_weight = [&](float) { return distribution(rng); };
    layer.weights = layer.weights.unaryExpr(generate_weight);
}

inline void layer_init_sigmoid(Layer &layer,
                               int size,
                               int input_size,
                               int batch_size,
                               std::minstd_rand &rng)
{
    layer_init_zero(layer, size, input_size, batch_size);

    const auto max_weight =
        4.0f * std::sqrt(6.0f / (static_cast<float>(input_size) +
                                 static_cast<float>(size)));
    std::uniform_real_distribution<float> distribution(-max_weight, max_weight);
    const auto generate_weight = [&](float) { return distribution(rng); };
    layer.weights = layer.weights.unaryExpr(generate_weight);
}

inline void layer_predict_linear(Layer &layer, const Eigen::MatrixXf &input)
{
    layer.activations = layer.biases.replicate(1, layer.activations.cols());
    layer.activations.noalias() += layer.weights * input;
}

inline void layer_predict_leaky_relu(Layer &layer, const Eigen::MatrixXf &input)
{
    layer_predict_linear(layer, input);
    layer.activations = layer.activations.cwiseMax(0.01f * layer.activations);
}

inline void layer_predict_sigmoid(Layer &layer, const Eigen::MatrixXf &input)
{
    layer_predict_linear(layer, input);
    layer.activations = 0.5f * (layer.activations.array() * 0.5f).tanh() + 0.5f;
}

inline void layer_update_deltas_leaky_relu(Layer &layer,
                                           const Layer &next_layer)
{
    layer.deltas.noalias() = next_layer.weights.transpose() * next_layer.deltas;
    layer.deltas.array() *=
        (layer.activations.array() > 0.0f).template cast<float>() * 0.99f +
        0.01f;
}

inline void layer_update_weights(Layer &layer,
                                 const Eigen::MatrixXf &input,
                                 int batch_size,
                                 float learning_rate)
{
    layer.weights.noalias() -= learning_rate / static_cast<float>(batch_size) *
                               layer.deltas * input.transpose();
    layer.biases.noalias() -= learning_rate / static_cast<float>(batch_size) *
                              layer.deltas.rowwise().sum();
}

inline void network_update_deltas(std::vector<Layer> &layers,
                                  const Eigen::MatrixXf &target_output,
                                  int batch_size)
{
    const auto &output = layers.back().activations.array();
    layers.back().deltas =
        (output - target_output.array()) * output * (1.0f - output);

    // If the batch size is smaller than the matrix size (for instance because
    // the size of the dataset is not a multiple of the batch size, and this is
    // the last batch of the dataset), set the inactive columns to zero to
    // ensure no contribution to the gradients. Note that we are still
    // performing all other computations as if the batch size was not reduced,
    // because we assume that this is the case most of the time (for a 100x100
    // image with 10'000 pixels and a batch size of 32, there would only be one
    // smaller batch size every 312 normal batches). Also, for a typical batch
    // size of 16 or 32, having to explicitely index a sub-range of columns
    // might actually reduce performance because of the vectorized
    // implementation.
    const auto num_inactive_cols = layers.back().deltas.cols() - batch_size;
    layers.back().deltas.rightCols(num_inactive_cols).setZero();

    for (std::size_t i {layers.size() - 1}; i > 0; --i)
    {
        layer_update_deltas_leaky_relu(layers[i - 1], layers[i]);
    }
}

inline void network_update_weights(std::vector<Layer> &layers,
                                   const Eigen::MatrixXf &input,
                                   int batch_size,
                                   float learning_rate)
{
    for (std::size_t i {layers.size() - 1}; i > 0; --i)
    {
        layer_update_weights(
            layers[i], layers[i - 1].activations, batch_size, learning_rate);
    }
    layer_update_weights(layers.front(), input, batch_size, learning_rate);
}

} // namespace

std::vector<Layer> network_init(const std::vector<int> &sizes,
                                int batch_size,
                                std::minstd_rand &rng)
{
    std::vector<Layer> layers(sizes.size() - 1);

    for (std::size_t i {0}; i < sizes.size() - 2; ++i)
    {
        layer_init_leaky_relu(
            layers[i], sizes[i + 1], sizes[i], batch_size, rng);
    }
    layer_init_sigmoid(
        layers.back(), sizes.back(), sizes[sizes.size() - 2], batch_size, rng);

    return layers;
}

void forward_pass(std::vector<Layer> &layers, const Eigen::MatrixXf &input)
{
    layer_predict_leaky_relu(layers.front(), input);
    for (std::size_t i {1}; i < layers.size() - 1; ++i)
    {
        layer_predict_leaky_relu(layers[i], layers[i - 1].activations);
    }
    layer_predict_sigmoid(layers.back(), layers[layers.size() - 2].activations);
}

void training_pass(std::vector<Layer> &layers,
                   const Eigen::MatrixXf &input,
                   const Eigen::MatrixXf &target_output,
                   int batch_size,
                   float learning_rate)
{
    forward_pass(layers, input);
    network_update_deltas(layers, target_output, batch_size);
    network_update_weights(layers, input, batch_size, learning_rate);
}
