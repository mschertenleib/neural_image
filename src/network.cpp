#include "network.hpp"

namespace
{

[[nodiscard]] inline auto relu(const auto &expr)
{
    return expr.array().max(0).matrix();
}

} // namespace

Network create_network(const std::vector<int> &hidden_layers_sizes)
{
    assert(!hidden_layers_sizes.empty());

    const auto num_hidden_layers = hidden_layers_sizes.size();

    Network network {};

    if (num_hidden_layers > 1)
    {
        network.weights_hidden_layers.resize(num_hidden_layers - 1);
        network.biases_hidden_layers.resize(num_hidden_layers - 1);
    }

    network.weights_first_layer.setRandom(hidden_layers_sizes.front(),
                                          Eigen::NoChange);
    network.biases_first_layer.setRandom(hidden_layers_sizes.front());

    for (std::size_t i {}; i < num_hidden_layers - 1; ++i)
    {
        network.weights_hidden_layers[i].setRandom(hidden_layers_sizes[i + 1],
                                                   hidden_layers_sizes[i]);
        network.biases_hidden_layers[i].setRandom(hidden_layers_sizes[i + 1]);
    }

    network.weights_last_layer.setRandom(Eigen::NoChange,
                                         hidden_layers_sizes.back());
    network.bias_last_layer.setRandom();

    return network;
}

float predict(const Network &network, float x, float y)
{
    const Eigen::Vector2f input(x, y);
    auto a =
        relu(network.weights_first_layer * input + network.biases_first_layer)
            .eval();
    for (std::size_t i {}; i < network.weights_hidden_layers.size(); ++i)
    {
        a = relu(network.weights_hidden_layers[i] * a +
                 network.biases_hidden_layers[i]);
    }
    // FIXME: the sum() call is just to get a scalar back, there is probably a
    // better way to do this
    const float output {
        (network.weights_last_layer * a + network.bias_last_layer)
            .array()
            .tanh()
            .sum()};
    return output;
}
