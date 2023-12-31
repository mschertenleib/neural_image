#include "network.hpp"

#include <cmath>
#include <random>

namespace
{

inline void
layer_init_zero(Layer &layer, Eigen::Index size, Eigen::Index input_size)
{
    layer.weights.setZero(size, input_size);
    layer.biases.setZero(size);
    layer.activations.setZero(size);
    layer.deltas.setZero(size);
}

inline void layer_init_leaky_relu(Layer &layer,
                                  Eigen::Index size,
                                  Eigen::Index input_size,
                                  std::minstd_rand &rng)
{
    layer_init_zero(layer, size, input_size);

    const auto std_dev = std::sqrt(2.0f / static_cast<float>(input_size));
    std::normal_distribution<float> distribution(0.0f, std_dev);
    const auto generate_weight = [&](float) { return distribution(rng); };
    layer.weights = layer.weights.unaryExpr(generate_weight);
}

inline void layer_init_sigmoid(Layer &layer,
                               Eigen::Index size,
                               Eigen::Index input_size,
                               std::minstd_rand &rng)
{
    layer_init_zero(layer, size, input_size);

    const auto max_weight =
        4.0f * std::sqrt(6.0f / (static_cast<float>(input_size) +
                                 static_cast<float>(size)));
    std::uniform_real_distribution<float> distribution(-max_weight, max_weight);
    const auto generate_weight = [&](float) { return distribution(rng); };
    layer.weights = layer.weights.unaryExpr(generate_weight);
}

inline void layer_predict_leaky_relu(Layer &layer, const Eigen::VectorXf &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = layer.activations.cwiseMax(0.01f * layer.activations);
}

inline void layer_predict_sigmoid(Layer &layer, const Eigen::VectorXf &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
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
                                 const Eigen::VectorXf &input,
                                 float learning_rate)
{
    layer.weights.noalias() -= learning_rate * layer.deltas * input.transpose();
    layer.biases.noalias() -= learning_rate * layer.deltas;
}

inline void network_update_deltas(std::vector<Layer> &layers,
                                  const Eigen::VectorXf &output)
{
    layers.back().deltas.array() =
        (layers.back().activations - output).array() *
        layers.back().activations.array() *
        (1.0f - layers.back().activations.array());

    for (std::size_t i {layers.size() - 1}; i > 0; --i)
    {
        layer_update_deltas_leaky_relu(layers[i - 1], layers[i]);
    }
}

} // namespace

void network_init(std::vector<Layer> &layers,
                  const std::vector<Eigen::Index> &sizes)
{
    layers.resize(sizes.size() - 1);

    std::random_device rd {};
    std::minstd_rand rng(rd());

    for (std::size_t i {0}; i < sizes.size() - 2; ++i)
    {
        layer_init_leaky_relu(layers[i], sizes[i + 1], sizes[i], rng);
    }
    layer_init_sigmoid(
        layers.back(), sizes.back(), sizes[sizes.size() - 2], rng);
}

void forward_pass(std::vector<Layer> &layers, const Eigen::VectorXf &input)
{
    layer_predict_leaky_relu(layers.front(), input);
    for (std::size_t i {1}; i < layers.size() - 1; ++i)
    {
        layer_predict_leaky_relu(layers[i], layers[i - 1].activations);
    }
    layer_predict_sigmoid(layers.back(), layers[layers.size() - 2].activations);
}

void backward_pass(std::vector<Layer> &layers,
                            const Eigen::VectorXf &input,
                            const Eigen::VectorXf &output,
                            float learning_rate)
{
    network_update_deltas(layers, output);

    for (std::size_t i {layers.size() - 1}; i > 0; --i)
    {
        layer_update_weights(
            layers[i], layers[i - 1].activations, learning_rate);
    }
    layer_update_weights(layers.front(), input, learning_rate);
}
