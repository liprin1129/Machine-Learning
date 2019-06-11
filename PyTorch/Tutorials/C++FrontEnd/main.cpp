#include <torch/torch.h>
#include <iostream>

// 1. Defining a Module and Registering Parameters
/*
import torch

class Net(torch.nn.Module):
  def __init__(self, N, M):
    super(Net, self).__init__()
    self.W = torch.nn.Parameter(torch.randn(N, M))
    self.b = torch.nn.Parameter(torch.randn(M))

  def forward(self, input):
    return torch.addmm(self.b, input, self.W)
*/

/*
class Net: public torch::nn::Module {
    public:
        torch::Tensor W, b;

        Net(int64_t N, int64_t M) {
            W = register_parameter("W", torch::randn({N, M}));
            b = register_parameter("b", torch::randn(M));
        }

        torch::Tensor forward(torch::Tensor input) {
            return torch::addmm(b, input, W);
        }
};
*/

// 2. Registering Submodules and Traversing the Module Hierarchy

/*
class Net(torch.nn.Module):
  def __init__(self, N, M):
      super(Net, self).__init__()
      # Registered as a submodule behind the scenes
      self.linear = torch.nn.Linear(N, M)
      self.another_bias = torch.nn.Parameter(torch.rand(M))

  def forward(self, input):
    return self.linear(input) + self.another_bias
*/

/*
class Net : public torch::nn::Module {
    public:
        //torch::nn::Linear linear{nullptr}; // For a module which has no default constructor, 
                                            // use = nullptr to give it the empty state 
                                            // (e.g. `Linear linear = nullptr;` instead of `Linear linear;`).
        torch::nn::Linear linear;
        torch::Tensor another_bias;

        Net(int64_t N, int64_t M) : linear(register_module("linear", torch::nn::Linear(N, M))) {
            //linear = register_module("linear", torch::nn::Linear(N, M));
            another_bias = register_parameter("b", torch::randn(M));
        }

        torch::Tensor forward(torch::Tensor input) {
            return linear(input);// + another_bias;
        }
};

int main() {
    Net net(4, 5);

    for (const auto& p : net.parameters()) {
        std::cout << p << std::endl;
    }

    for (const auto& pair : net.named_parameters()) {
        std::cout << pair.key() << ": " << pair.value() << std::endl;
    }

    std::cout << net.forward(torch::ones({2, 4})) << std::endl;
}
*/

using namespace torch;

// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 100;
/*
// using Sequential
nn::Sequential generator(
    // Layer 1
    nn::Conv2d(nn::Conv2dOptions(kNoiseSize, 256, 4).with_bias(false).transposed(true)),
    nn::BatchNorm(256),
    nn::Functional(torch::relu),
    // Layer 2
    nn::Conv2d(nn::Conv2dOptions(256, 128, 3)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
    nn::BatchNorm(128),
    nn::Functional(torch::relu),
    // Layer 3
    nn::Conv2d(nn::Conv2dOptions(128, 64, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
    nn::BatchNorm(64),
    nn::Functional(torch::relu),
    // Layer 4
    nn::Conv2d(nn::Conv2dOptions(64, 1, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
    nn::Functional(torch::tanh));

nn::Sequential discriminator(
    // Layer 1
    nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).with_bias(false)),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 2
    nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).with_bias(false)),
    nn::BatchNorm(128),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 3
    nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).with_bias(false)),
    nn::BatchNorm(256),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 4
    nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).with_bias(false)),
    nn::Functional(torch::sigmoid)
);
*/
// Explicitly pass inputs (in a functional way) between modules in the forward() method of a module we define ourselves

struct DCGeneratorImpl : torch::nn::Module {
    DCGeneratorImpl():
        conv1(nn::Conv2dOptions(kNoiseSize, 512, 4)
                .with_bias(false)
                .transposed(true)),
        batch_norm1(512),
        conv2(nn::Conv2dOptions(512, 256, 4)
                .stride(2)
                .padding(1)
                .with_bias(false)
                .transposed(true)),
        batch_norm2(256),
        conv3(nn::Conv2dOptions(256, 128, 4)
                .stride(2)
                .padding(1)
                .with_bias(false)
                .transposed(true)),
        batch_norm3(128),
        conv4(nn::Conv2dOptions(128, 64, 4)
                .stride(2)
                .padding(1)
                .with_bias(false)
                .transposed(true)),
        batch_norm4(64),
        conv5(nn::Conv2dOptions(64, 1, 4)
                .stride(2)
                .padding(1)
                .with_bias(false)
                .transposed(true)) {}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::relu(batch_norm4(conv4(x)));
        x = torch::tanh(conv5(x));
        return x;
    }

    nn::Conv2d conv1, conv2, conv3, conv4, conv5;
    nn::BatchNorm batch_norm1, batch_norm2, batch_norm3, batch_norm4;
};
TORCH_MODULE(DCGenerator);

struct DCDescriminatorImpl : torch::nn::Module {
    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::BatchNorm batch_norm2, batch_norm3, batch_norm4;

    DCDescriminatorImpl() :
        conv1(nn::Conv2dOptions(1, 64, 4)
                .stride(2)
                .padding(1)
                .with_bias(false)),
        conv2(nn::Conv2dOptions(64, 128, 4)
                .stride(2)
                .padding(1)
                .with_bias(false)),
        batch_norm2(128),
        conv3(nn::Conv2dOptions(128, 256, 4)
                .stride(1)
                .padding(1)
                .with_bias(false)),
        batch_norm3(256),
        conv4(nn::Conv2dOptions(256, 1, 3)
                .stride(1)
                .padding(0)
                .with_bias(false)){}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::leaky_relu(conv1(x), 0.2);
        x = torch::leaky_relu(batch_norm2(conv2(x), 0.2));
        x = torch::leaky_relu(batch_norm3(conv2(x), 0.2));
        x = torch::sigmoid(conv4(x));
        return x;
    }
};
TORCH_MODULE(DCDescriminator);

int main() {
  auto dataset = torch::data::datasets::MNIST(".mnist")
                    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                    .map(torch::data::transforms::Stack<>());

  //Net net;
  DCGenerator generator;
}