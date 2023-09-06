#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Dense>

#include <vector>

struct Network
{
    Eigen::MatrixX2f weights_first_layer;
    Eigen::VectorXf biases_first_layer;
    std::vector<Eigen::MatrixXf> weights_hidden_layers;
    std::vector<Eigen::VectorXf> biases_hidden_layers;
    Eigen::RowVectorXf weights_last_layer;
    Eigen::Vector<float, 1> bias_last_layer;
};

[[nodiscard]] Network
create_network(const std::vector<int> &hidden_layers_sizes);

[[nodiscard]] float predict(const Network &network, float x, float y);

#endif // NETWORK_HPP
