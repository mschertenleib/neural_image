#include "network.hpp"

#include <cmath>
#include <random>

namespace
{

inline void layer_init_zero(Layer &layer, int size, int input_size)
{
    layer.weights.setZero(size, input_size);
    layer.biases.setZero(size);
    layer.activations.setZero(size);
    layer.deltas.setZero(size);
}

inline void layer_init_leaky_relu(Layer &layer,
                                  int size,
                                  int input_size,
                                  std::minstd_rand &rng)
{
    layer_init_zero(layer, size, input_size);

    const auto std_dev = std::sqrt(2.0f / static_cast<float>(input_size));
    std::normal_distribution<float> distribution(0.0f, std_dev);
    const auto generate_weight = [&](float) { return distribution(rng); };
    layer.weights = layer.weights.unaryExpr(generate_weight);
}

inline void layer_init_sigmoid(Layer &layer,
                               int size,
                               int input_size,
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

inline void network_update_deltas(Network &network,
                                  const Eigen::VectorXf &output)
{
    network.layers.back().deltas.array() =
        (network.layers.back().activations - output).array() *
        network.layers.back().activations.array() *
        (1.0f - network.layers.back().activations.array());

    for (std::size_t i {network.layers.size() - 1}; i > 0; --i)
    {
        layer_update_deltas_leaky_relu(network.layers[i - 1],
                                       network.layers[i]);
    }
}

} // namespace

void network_init(Network &network, const std::vector<unsigned int> &sizes)
{
    network.layers.resize(sizes.size());

    std::random_device rd {};
    std::minstd_rand rng(rd());

    for (std::size_t i {0}; i < sizes.size() - 1; ++i)
    {
        layer_init_leaky_relu(network.layers[i],
                              static_cast<int>(sizes[i + 1]),
                              static_cast<int>(sizes[i]),
                              rng);
    }
    layer_init_sigmoid(
        network.layers.back(), 3, static_cast<int>(sizes.back()), rng);
}

void network_predict(Network &network, const Eigen::VectorXf &input)
{
    layer_predict_leaky_relu(network.layers.front(), input);
    for (std::size_t i {1}; i < network.layers.size() - 1; ++i)
    {
        layer_predict_leaky_relu(network.layers[i],
                                 network.layers[i - 1].activations);
    }
    layer_predict_sigmoid(
        network.layers.back(),
        network.layers[network.layers.size() - 2].activations);
}

void network_update_weights(Network &network,
                            const Eigen::VectorXf &input,
                            const Eigen::VectorXf &output,
                            float learning_rate)
{
    network_update_deltas(network, output);

    for (std::size_t i {network.layers.size() - 1}; i > 0; --i)
    {
        layer_update_weights(network.layers[i],
                             network.layers[i - 1].activations,
                             learning_rate);
    }
    layer_update_weights(network.layers.front(), input, learning_rate);
}
