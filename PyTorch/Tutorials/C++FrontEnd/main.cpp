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
const int64_t kBatchSize = 64;
const int64_t kNumberOfEpochs = 30;
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

int main() {
    torch::manual_seed(1);
    
    auto dataset = torch::data::datasets::MNIST("./mnist")
                    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                    .map(torch::data::transforms::Stack<>());

    const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(8));
    
    /* // Print data and label
    for (torch::data::Example<>& batch : *data_loader) {
        std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
        for (int64_t i = 0; i < batch.data.size(0); ++i) {
            std::cout << batch.target[i].item<int64_t>() << " ";
        }
        std::cout << std::endl;
    }
    */

    // Create the device we pass around based on whether CUDA is available.
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
    }
    
    struct DCGeneratorImpl : torch::nn::Module {
        DCGeneratorImpl():
            conv1(nn::Conv2dOptions(kNoiseSize, 256, 4)
                    .with_bias(false)
                    .transposed(true)),
            batch_norm1(256),
            conv2(nn::Conv2dOptions(256, 128, 3)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
            batch_norm2(128),
            conv3(nn::Conv2dOptions(128, 64, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
            batch_norm3(64),
            conv4(nn::Conv2dOptions(64, 1, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)) {
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("conv3", conv3);
            register_module("conv4", conv4);

            register_module("batch_norm1", batch_norm1);
            register_module("batch_norm2", batch_norm2);
            register_module("batch_norm3", batch_norm3);
            }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(batch_norm1(conv1(x)));
            x = torch::relu(batch_norm2(conv2(x)));
            x = torch::relu(batch_norm3(conv3(x)));
            x = torch::tanh(conv4(x));
            return x;
        }

        nn::Conv2d conv1, conv2, conv3, conv4;
        nn::BatchNorm batch_norm1, batch_norm2, batch_norm3;
    };
    TORCH_MODULE(DCGenerator);

    struct DCDiscriminatorImpl : torch::nn::Module {
        DCDiscriminatorImpl() :
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
                    .stride(2)
                    .padding(1)
                    .with_bias(false)),
            batch_norm3(256),
            conv4(nn::Conv2dOptions(256, 1, 3)
                    .stride(1)
                    .padding(0)
                    .with_bias(false)){
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("conv3", conv3);
            register_module("conv4", conv4);

            register_module("batch_norm2", batch_norm2);
            register_module("batch_norm3", batch_norm3);
                    }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::leaky_relu(conv1(x), 0.2);
            x = torch::leaky_relu(batch_norm2(conv2(x)), 0.2);
            x = torch::leaky_relu(batch_norm3(conv3(x)), 0.2);
            x = torch::sigmoid(conv4(x));
            return x;
        }

        nn::Conv2d conv1, conv2, conv3, conv4;
        nn::BatchNorm batch_norm2, batch_norm3;
    };
    TORCH_MODULE(DCDiscriminator);

    DCGenerator generator;
    generator->to(device);
    DCDiscriminator discriminator;
    discriminator->to(device);


    /*
    nn::Sequential generator(
        // Layer 1
        nn::Conv2d(nn::Conv2dOptions(kNoiseSize, 256, 4)
                        .with_bias(false)
                        .transposed(true)),
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
    generator->to(device);

    nn::Sequential discriminator(
        // Layer 1
        nn::Conv2d(
            nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).with_bias(false)),
        nn::Functional(torch::leaky_relu, 0.2),
        // Layer 2
        nn::Conv2d(
            nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).with_bias(false)),
        nn::BatchNorm(128),
        nn::Functional(torch::leaky_relu, 0.2),
        // Layer 3
        nn::Conv2d(
            nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).with_bias(false)),
        nn::BatchNorm(256),
        nn::Functional(torch::leaky_relu, 0.2),
        // Layer 4
        nn::Conv2d(
            nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).with_bias(false)),
        nn::Functional(torch::sigmoid));
    discriminator->to(device);
    */

    torch::optim::Adam generator_optimizer(
        generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
    torch::optim::Adam discriminator_optimizer(
        discriminator->parameters(), torch::optim::AdamOptions(5e-4).beta1(0.5));

    
    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        int64_t batch_index = 0;

        for (torch::data::Example<>& batch : *data_loader) {
            // Train discriminator with real images.
            discriminator->zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
            torch::Tensor real_output = discriminator->forward(real_images);
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();

            //std::printf("\r[%2ld/%2ld][%3ld/%3ld]", epoch, kNumberOfEpochs, ++batch_index, batches_per_epoch);
            
            // Train discriminator with fake images.
            torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
            torch::Tensor fake_images = generator->forward(noise);
            torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
            torch::Tensor fake_output = discriminator->forward(fake_images.detach());
            torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();

            torch::Tensor d_loss = d_loss_real + d_loss_fake;
            discriminator_optimizer.step();

            // Train generator
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();

            std::printf(
                "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
                epoch,
                kNumberOfEpochs,
                ++batch_index,
                batches_per_epoch,
                d_loss.item<float>(),
                g_loss.item<float>());
        }
    }
}