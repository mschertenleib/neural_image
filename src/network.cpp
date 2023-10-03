#include "network.hpp"

#include <cmath>
#include <random>

namespace
{

template <int Size, int Input_size, Activation A>
void layer_init_zero(Layer<Size, Input_size, A> &layer,
                     int size,
                     int input_size)
{
    layer.weights.setZero(size, input_size);
    layer.biases.setZero(size);
    layer.activations.setZero(size);
    layer.deltas.setZero(size);
}

template <int Size, int Input_size>
void layer_init(Layer<Size, Input_size, Activation::relu> &layer,
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

template <int Size, int Input_size>
void layer_init(Layer<Size, Input_size, Activation::leaky_relu> &layer,
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

template <int Size, int Input_size>
void layer_init(Layer<Size, Input_size, Activation::sigmoid> &layer,
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

template <int Size, int Input_size>
inline void layer_predict(Layer<Size, Input_size, Activation::relu> &layer,
                          const Eigen::Vector<float, Input_size> &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = layer.activations.cwiseMax(0.0f);
}

template <int Size, int Input_size>
inline void
layer_predict(Layer<Size, Input_size, Activation::leaky_relu> &layer,
              const Eigen::Vector<float, Input_size> &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = layer.activations.cwiseMax(0.01f * layer.activations);
}

template <int Size, int Input_size>
inline void layer_predict(Layer<Size, Input_size, Activation::sigmoid> &layer,
                          const Eigen::Vector<float, Input_size> &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = 0.5f * (layer.activations.array() * 0.5f).tanh() + 0.5f;
}

template <int Size, int Input_size, int Size_next, Activation A_next>
inline void
layer_update_deltas(Layer<Size, Input_size, Activation::relu> &layer,
                    const Layer<Size_next, Size, A_next> &next_layer)
{
    layer.deltas.noalias() = next_layer.weights.transpose() * next_layer.deltas;
    layer.deltas.array() *=
        (layer.activations.array() > 0.0f).template cast<float>();
}

template <int Size, int Input_size, int Size_next, Activation A_next>
inline void
layer_update_deltas(Layer<Size, Input_size, Activation::leaky_relu> &layer,
                    const Layer<Size_next, Size, A_next> &next_layer)
{
    layer.deltas.noalias() = next_layer.weights.transpose() * next_layer.deltas;
    layer.deltas.array() *=
        (layer.activations.array() > 0.0f).template cast<float>() * 0.99f +
        0.01f;
}

template <int Size, int Input_size, Activation A>
inline void layer_update_weights(Layer<Size, Input_size, A> &layer,
                                 const Eigen::Vector<float, Input_size> &input,
                                 float learning_rate)
{
    layer.weights.noalias() -= learning_rate * layer.deltas * input.transpose();
    layer.biases.noalias() -= learning_rate * layer.deltas;
}

void network_update_deltas(Network &network, const Network::Output &output)
{
    network.output_layer.deltas.array() =
        (network.output_layer.activations - output).array() *
        network.output_layer.activations.array() *
        (1.0f - network.output_layer.activations.array());

    layer_update_deltas(network.hidden_layers.back(), network.output_layer);
    for (std::size_t i {network.hidden_layers.size() - 1}; i > 0; --i)
    {
        layer_update_deltas(network.hidden_layers[i - 1],
                            network.hidden_layers[i]);
    }
}

} // namespace

void network_init(Network &network, const std::vector<int> &sizes)
{
    assert(sizes.size() >= 2);

    const std::size_t num_hidden_layers {sizes.size() - 1};
    network.hidden_layers.resize(num_hidden_layers);

    std::random_device rd {};
    std::minstd_rand rng(rd());

    for (std::size_t i {0}; i < num_hidden_layers; ++i)
    {
        layer_init(network.hidden_layers[i], sizes[i + 1], sizes[i], rng);
    }
    layer_init(network.output_layer, 3, sizes.back(), rng);
}

void network_predict(Network &network, const Network::Input &input)
{
    layer_predict(network.hidden_layers.front(), input);
    for (std::size_t i {1}; i < network.hidden_layers.size(); ++i)
    {
        layer_predict(network.hidden_layers[i],
                      network.hidden_layers[i - 1].activations);
    }
    layer_predict(network.output_layer,
                  network.hidden_layers.back().activations);
}

void network_update_weights(Network &network,
                            const Network::Input &input,
                            const Network::Output &output,
                            float learning_rate)
{
    network_update_deltas(network, output);

    layer_update_weights(network.output_layer,
                         network.hidden_layers.back().activations,
                         learning_rate);
    for (std::size_t i {network.hidden_layers.size() - 1}; i > 0; --i)
    {
        layer_update_weights(network.hidden_layers[i],
                             network.hidden_layers[i - 1].activations,
                             learning_rate);
    }
    layer_update_weights(network.hidden_layers.front(), input, learning_rate);
}
