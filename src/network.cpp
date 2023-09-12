#include "network.hpp"

namespace
{

template <int size, int previous_layer_size>
void layer_init(Layer<size, previous_layer_size> &layer,
                int runtime_size,
                int runtime_previous_layer_size)
{
    layer.weights.setRandom(runtime_size, runtime_previous_layer_size);
    layer.biases.setRandom(runtime_size);
    layer.activations.setZero(runtime_size);
}

template <int size, int previous_layer_size>
inline void
hidden_layer_predict(Layer<size, previous_layer_size> &layer,
                     const Eigen::Vector<float, previous_layer_size> &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = layer.activations.array().max(0).matrix();
}

template <int size, int previous_layer_size>
inline void
output_layer_predict(Layer<size, previous_layer_size> &layer,
                     const Eigen::Vector<float, previous_layer_size> &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = layer.activations.array().tanh().matrix();
}

} // namespace

void network_init(Network &network, const std::vector<int> &hidden_layers_sizes)
{
    assert(!hidden_layers_sizes.empty());

    const auto num_hidden_layers = hidden_layers_sizes.size();
    if (num_hidden_layers > 1)
    {
        network.additional_hidden_layers.resize(num_hidden_layers - 1);
    }

    layer_init(network.first_hidden_layer, hidden_layers_sizes.front(), 2);

    for (std::size_t i {0}; i < num_hidden_layers - 1; ++i)
    {
        layer_init(network.additional_hidden_layers[i],
                   hidden_layers_sizes[i + 1],
                   hidden_layers_sizes[i]);
    }

    layer_init(network.output_layer, 3, hidden_layers_sizes.back());
}

Eigen::Vector3f network_predict(Network &network, float x, float y)
{
    hidden_layer_predict(network.first_hidden_layer, Eigen::Vector2f(x, y));

    if (network.additional_hidden_layers.empty())
    {
        output_layer_predict(network.output_layer,
                             network.first_hidden_layer.activations);
    }
    else
    {
        hidden_layer_predict(network.additional_hidden_layers.front(),
                             network.first_hidden_layer.activations);
        for (std::size_t i {1}; i < network.additional_hidden_layers.size();
             ++i)
        {
            hidden_layer_predict(
                network.additional_hidden_layers[i],
                network.additional_hidden_layers[i - 1].activations);
        }
        output_layer_predict(
            network.output_layer,
            network.additional_hidden_layers.back().activations);
    }

    return network.output_layer.activations;
}
