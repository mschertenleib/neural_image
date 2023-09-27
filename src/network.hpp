#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Dense>

#include <vector>

enum struct Activation
{
    leaky_relu,
    sigmoid
};

template <int Size, int Input_size, Activation A>
struct Layer
{
    Eigen::Matrix<float, Size, Input_size> weights;
    Eigen::Vector<float, Size> biases;
    Eigen::Vector<float, Size> activations;
    Eigen::Vector<float, Size> deltas;
};

struct Network
{
    using Input = Eigen::VectorXf;
    using Output = Eigen::Vector3f;

    std::vector<Layer<Eigen::Dynamic, Eigen::Dynamic, Activation::leaky_relu>>
        hidden_layers;
    Layer<3, Eigen::Dynamic, Activation::sigmoid> output_layer;
};

void network_init(Network &network, const std::vector<int> &sizes);

void network_predict(Network &network, const Network::Input &input);

void network_update_weights(Network &network,
                            const Network::Input &input,
                            const Network::Output &output,
                            float learning_rate);

#endif // NETWORK_HPP
