#include "network.hpp"

#include <cmath>
#include <random>

namespace
{

template <int N, int N_previous, Activation A>
void init_layer_zero(Layer<N, N_previous, A> &layer,
                     int size,
                     int previous_layer_size)
{
    layer.weights.setZero(size, previous_layer_size);
    layer.biases.setZero(size);
    layer.activations.setZero(size);
    layer.deltas.setZero(size);
}

template <int N, int N_previous>
void init_layer_leaky_relu(Layer<N, N_previous, Activation::leaky_relu> &layer,
                           int size,
                           int previous_layer_size,
                           std::minstd_rand &rng)
{
    init_layer_zero(layer, size, previous_layer_size);

    const auto std_dev =
        std::sqrt(2.0f / static_cast<float>(previous_layer_size));
    std::normal_distribution<float> distribution(0.0f, std_dev);
    const auto generate_weight = [&](float) { return distribution(rng); };
    layer.weights = layer.weights.unaryExpr(generate_weight);
}

template <int N, int N_previous>
void init_layer_sigmoid(Layer<N, N_previous, Activation::sigmoid> &layer,
                        int size,
                        int previous_layer_size)
{
    init_layer_zero(layer, size, previous_layer_size);

    const auto max_weight =
        4.0f * std::sqrt(6.0f / (static_cast<float>(previous_layer_size) +
                                 static_cast<float>(size)));
    layer.weights.setRandom();
    layer.weights *= max_weight;
}

template <int N, int N_previous>
inline void
predict_leaky_relu(Layer<N, N_previous, Activation::leaky_relu> &layer,
                   const Eigen::Vector<float, N_previous> &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = layer.activations.cwiseMax(0.01f * layer.activations);
}

template <int N, int N_previous>
inline void predict_sigmoid(Layer<N, N_previous, Activation::sigmoid> &layer,
                            const Eigen::Vector<float, N_previous> &input)
{
    layer.activations = layer.biases;
    layer.activations.noalias() += layer.weights * input;
    layer.activations = 0.5f * (layer.activations.array() * 0.5f).tanh() + 0.5f;
}

template <int N, int N_previous, int N_next, Activation A_next>
inline void
compute_deltas_leaky_relu(Layer<N, N_previous, Activation::leaky_relu> &layer,
                          const Layer<N_next, N, A_next> &next_layer)
{
    layer.deltas.noalias() = next_layer.weights.transpose() * next_layer.deltas;
    layer.deltas.array() *=
        (layer.activations.array() > 0.0f).template cast<float>() * 0.99f +
        0.01f;
}

void compute_deltas(Network &network, const Eigen::Vector3f &output)
{
    network.output_layer.deltas.array() =
        (network.output_layer.activations - output).array() *
        network.output_layer.activations.array() *
        (1.0f - network.output_layer.activations.array());

    compute_deltas_leaky_relu(network.hidden_layers.back(),
                              network.output_layer);
    for (std::size_t i {network.hidden_layers.size() - 1}; i > 0; --i)
    {
        compute_deltas_leaky_relu(network.hidden_layers[i - 1],
                                  network.hidden_layers[i]);
    }
}

template <int N, int N_previous, Activation A>
inline void update_weights(
    Layer<N, N_previous, A> &layer,
    const Eigen::Vector<float, N_previous> &previous_layer_activations,
    float learning_rate)
{
    layer.weights.noalias() -=
        learning_rate * layer.deltas * previous_layer_activations.transpose();
    layer.biases.noalias() -= learning_rate * layer.deltas;
}

void update_weights(Network &network,
                    const Eigen::VectorXf &input,
                    float learning_rate)
{
    update_weights(network.output_layer,
                   network.hidden_layers.back().activations,
                   learning_rate);
    for (std::size_t i {network.hidden_layers.size() - 1}; i > 0; --i)
    {
        update_weights(network.hidden_layers[i],
                       network.hidden_layers[i - 1].activations,
                       learning_rate);
    }
    update_weights(network.hidden_layers.front(), input, learning_rate);
}

} // namespace

void init_network(Network &network, const std::vector<int> &sizes)
{
    assert(sizes.size() >= 2);
    assert(sizes.front() == 2);

    const std::size_t num_hidden_layers {sizes.size() - 1};
    network.hidden_layers.resize(num_hidden_layers);

    std::random_device rd {};
    std::minstd_rand rng(rd());

    for (std::size_t i {0}; i < num_hidden_layers; ++i)
    {
        init_layer_leaky_relu(
            network.hidden_layers[i], sizes[i + 1], sizes[i], rng);
    }
    init_layer_sigmoid(network.output_layer, 3, sizes.back());
}

void predict(Network &network, const Eigen::VectorXf &input)
{
    predict_leaky_relu(network.hidden_layers.front(), input);
    for (std::size_t i {1}; i < network.hidden_layers.size(); ++i)
    {
        predict_leaky_relu(network.hidden_layers[i],
                           network.hidden_layers[i - 1].activations);
    }
    predict_sigmoid(network.output_layer,
                    network.hidden_layers.back().activations);
}

void stochastic_gradient_descent(Network &network,
                                 const Eigen::VectorXf &input,
                                 const Eigen::Vector3f &output,
                                 float learning_rate)
{
    predict(network, input);
    compute_deltas(network, output);
    update_weights(network, input, learning_rate);
}
