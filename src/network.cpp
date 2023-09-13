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
    layer.deltas.setZero(runtime_size);
}

template <int size, int previous_layer_size>
inline void
hidden_layer_predict(Layer<size, previous_layer_size> &layer,
                     const Eigen::Vector<float, previous_layer_size> &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = layer.activations.array().tanh().matrix();
    //layer.activations = layer.activations.cwiseMax(0.0f);
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

template <int size, int previous_layer_size, int next_layer_size>
inline void compute_deltas(Layer<size, previous_layer_size> &layer,
                           const Layer<next_layer_size, size> &next_layer)
{
    layer.deltas.noalias() = next_layer.weights.transpose() * next_layer.deltas;

    layer.deltas = layer.deltas.cwiseProduct(
        (1.0f - layer.activations.array().square()).matrix());

    // FIXME: there must be a way to do this without multiplying by 0 or 1
    //layer.deltas = layer.deltas.cwiseProduct(
    //    (layer.activations.array() > 0.0f).matrix().template cast<float>());
}

void compute_deltas(Network &network, const Eigen::Vector3f &output)
{
    network.output_layer.deltas =
        (network.output_layer.activations - output)
            .cwiseProduct(
                (1.0f - network.output_layer.activations.array().square())
                    .matrix());

    if (network.additional_hidden_layers.empty())
    {
        compute_deltas(network.first_hidden_layer, network.output_layer);
    }
    else
    {
        compute_deltas(network.additional_hidden_layers.back(),
                       network.output_layer);
        for (std::size_t i {network.additional_hidden_layers.size() - 1}; i > 0;
             --i)
        {
            compute_deltas(network.additional_hidden_layers[i - 1],
                           network.additional_hidden_layers[i]);
        }
        compute_deltas(network.first_hidden_layer,
                       network.additional_hidden_layers.front());
    }
}

template <int size, int previous_layer_size>
inline void update_weights(
    Layer<size, previous_layer_size> &layer,
    const Eigen::Vector<float, previous_layer_size> &previous_layer_activations,
    float learning_rate)
{
    layer.weights.noalias() -=
        learning_rate * layer.deltas * previous_layer_activations.transpose();
    layer.biases.noalias() -= learning_rate * layer.deltas;
}

void update_weights(Network &network,
                    const Eigen::Vector2f &input,
                    float learning_rate)
{
    if (network.additional_hidden_layers.empty())
    {
        update_weights(network.output_layer,
                       network.first_hidden_layer.activations,
                       learning_rate);
    }
    else
    {
        update_weights(network.output_layer,
                       network.additional_hidden_layers.back().activations,
                       learning_rate);
        for (std::size_t i {network.additional_hidden_layers.size() - 1}; i > 0;
             --i)
        {
            update_weights(network.additional_hidden_layers[i],
                           network.additional_hidden_layers[i - 1].activations,
                           learning_rate);
        }
        update_weights(network.additional_hidden_layers.front(),
                       network.first_hidden_layer.activations,
                       learning_rate);
    }

    update_weights(network.first_hidden_layer, input, learning_rate);
}

[[nodiscard]] float loss(const Network &network, const Eigen::Vector3f &output)
{
    return 0.5f *
           (output - network.output_layer.activations).array().square().sum();
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

void network_predict(Network &network, const Eigen::Vector2f &input)
{
    hidden_layer_predict(network.first_hidden_layer, input);

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
}

void stochastic_gradient_descent(Network &network,
                                 const Eigen::Vector2f &input,
                                 const Eigen::Vector3f &output,
                                 float learning_rate)
{
    network_predict(network, input);
    compute_deltas(network, output);
    update_weights(network, input, learning_rate);
}
