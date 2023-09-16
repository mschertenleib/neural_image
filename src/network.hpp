#ifndef NETWORK_HPP
#define NETWORK_HPP

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>

#include <vector>

enum struct Activation
{
    leaky_relu,
    sigmoid
};

template <int N, int N_previous, Activation A>
struct Layer
{
    Eigen::Matrix<float, N, N_previous> weights;
    Eigen::Vector<float, N> biases;
    Eigen::Vector<float, N> activations;
    Eigen::Vector<float, N> deltas;
};

struct Network
{
    std::vector<Layer<Eigen::Dynamic, Eigen::Dynamic, Activation::leaky_relu>>
        hidden_layers;
    Layer<3, Eigen::Dynamic, Activation::sigmoid> output_layer;
};

void init_network(Network &network, const std::vector<int> &sizes);

void predict(Network &network, const Eigen::VectorXf &input);

// TODO: mini-batch gradient descent
void stochastic_gradient_descent(Network &network,
                                 const Eigen::VectorXf &input,
                                 const Eigen::Vector3f &output,
                                 float learning_rate);

#endif // NETWORK_HPP
