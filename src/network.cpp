#include "network.hpp"

namespace
{

template <int Layer_size, int Previous_layer_size>
void layer_init(Layer<Layer_size, Previous_layer_size> &layer,
                int layer_size,
                int previous_layer_size)
{
    layer.weights.setRandom(layer_size, previous_layer_size);
    layer.biases.setRandom(layer_size);
    layer.activations.setZero(layer_size);
}

template <int layer_size, int previous_layer_size>
inline void
hidden_layer_predict(Layer<layer_size, previous_layer_size> &layer,
                     const Eigen::Vector<float, previous_layer_size> &input)
{
    layer.activations =
        (layer.weights * input + layer.biases).array().max(0).matrix();
}

template <int layer_size, int previous_layer_size>
inline void
output_layer_predict(Layer<layer_size, previous_layer_size> &layer,
                     const Eigen::Vector<float, previous_layer_size> &input)
{
    layer.activations =
        (layer.weights * input + layer.biases).array().tanh().matrix();
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
    if (network.additional_hidden_layers.empty())
    {
        hidden_layer_predict(network.first_hidden_layer, Eigen::Vector2f(x, y));
        output_layer_predict(network.output_layer,
                             network.first_hidden_layer.activations);
    }
    else
    {
        hidden_layer_predict(network.first_hidden_layer, Eigen::Vector2f(x, y));
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
