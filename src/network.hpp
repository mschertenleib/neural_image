#ifndef NETWORK_HPP
#define NETWORK_HPP

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>

#include <vector>

template <int size, int previous_layer_size>
struct Layer
{
    Eigen::Matrix<float, size, previous_layer_size> weights;
    Eigen::Vector<float, size> biases;
    Eigen::Vector<float, size> activations;
    Eigen::Vector<float, size> deltas;
};

struct Network
{
    Layer<Eigen::Dynamic, 2> first_hidden_layer;
    std::vector<Layer<Eigen::Dynamic, Eigen::Dynamic>> additional_hidden_layers;
    Layer<3, Eigen::Dynamic> output_layer;
};

void network_init(Network &network,
                  const std::vector<int> &hidden_layers_sizes);

void network_predict(Network &network, const Eigen::Vector2f &input);

void stochastic_gradient_descent(Network &network,
                                 const Eigen::Vector2f &input,
                                 const Eigen::Vector3f &output,
                                 float learning_rate);

#endif // NETWORK_HPP
