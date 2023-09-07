#ifndef NETWORK_HPP
#define NETWORK_HPP

#define EIGEN_NO_AUTOMATIC_RESIZING
#include <Eigen/Dense>

#include <vector>

template <int layer_size, int previous_layer_size>
struct Layer
{
    Eigen::Matrix<float, layer_size, previous_layer_size> weights;
    Eigen::Vector<float, layer_size> biases;
    Eigen::Vector<float, layer_size> activations;
};

struct Network
{
    Layer<Eigen::Dynamic, 2> first_hidden_layer;
    std::vector<Layer<Eigen::Dynamic, Eigen::Dynamic>> additional_hidden_layers;
    Layer<3, Eigen::Dynamic> output_layer;
};

void network_init(Network &network,
                  const std::vector<int> &hidden_layers_sizes);

[[nodiscard]] Eigen::Vector3f
network_predict(Network &network, float x, float y);

#endif // NETWORK_HPP
