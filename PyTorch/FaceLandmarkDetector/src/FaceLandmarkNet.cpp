#include "FaceLandmarkNet.h"

// ************************ //
// * FaceLandmarNet class * //
// ************************ //

FaceLandmarkNetImpl::FaceLandmarkNetImpl(int inputChannel, bool verbose) {
    //std::cout << "Constructor" << std::endl;
    
    _verbose = verbose;
    //int inputChannel = 3;

    // Convolutional layer #1: 
    // [channel: input -> 32], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannel, 32, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm1 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(32));
    
    // Convolutional layer #2: 
    // [channel: 32 -> 64], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm2 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(64));

    // Convolutional layer #3: 
    // [channel: 64 -> 64], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm3 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(64));

    // Convolutional layer #4: 
    // [channel: 64 -> 64], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm4 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(64));

    // Convolutional layer #5: 
    // [channel: 64 -> 64], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm5 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(64));

    // Convolutional layer #6: 
    // [channel: 64 -> 128], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv6 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm6 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(128));

    // Convolutional layer #7: 
    // [channel: 128 -> 256], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv7 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm7 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(128));

    // Convolutional layer #7: 
    // [channel: 128 -> 128], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv8 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm8 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(256));

    // [1 x 1] Convolutional layer #8:
    // [channel: 256 -> 68], [filter: 1x1], [stride: 1x1], [padding:0]
    convOneXOne1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 136, 1)
                .stride(1)
                .padding(0)
                .with_bias(true));

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("conv5", conv5);
    register_module("conv6", conv6);
    register_module("conv7", conv7);
    register_module("conv8", conv8);
    register_module("convOneXOne1", convOneXOne1);
    
    
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
    register_module("batch_norm4", batch_norm4);
    register_module("batch_norm5", batch_norm5);
    register_module("batch_norm6", batch_norm6);
    register_module("batch_norm7", batch_norm7);
    register_module("batch_norm8", batch_norm8);
    
}

torch::Tensor FaceLandmarkNetImpl::forward(torch::Tensor x) {

    // ==========
    // Layer #1
    // ==========
    if (_verbose) std::cout << "Layer #1:\n";
    if (_verbose) std::cout << "\t Input: \t" << x.sizes() << std::endl;

    x = torch::relu(conv1(x));
    if (_verbose) std::cout << "\t Conv1: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;
    
    // ==========
    // Layer #2
    // ==========
    if (_verbose) std::cout << "Layer #2:\n";
    
    x = torch::relu(conv2(x));
    if (_verbose) std::cout << "\t Conv2: \t" << x.sizes() << std::endl;
    
    x = torch::relu(conv3(x));
    if (_verbose) std::cout << "\t Conv3: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // ==========
    // Layer #3
    // ==========
    if (_verbose) std::cout << "Layer #3:\n";
    
    x = torch::relu(conv4(x));
    if (_verbose) std::cout << "\t Conv4: \t" << x.sizes() << std::endl;
    
    x = torch::relu(conv5(x));
    if (_verbose) std::cout << "\t Conv5: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // ==========
    // Layer #4
    // ==========
    if (_verbose) std::cout << "Layer #4:\n";
    
    x = torch::relu(conv6(x));
    if (_verbose) std::cout << "\t Conv6: \t" << x.sizes() << std::endl;
    
    x = torch::relu(conv7(x));
    if (_verbose) std::cout << "\t Conv7: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // ==========
    // Layer #5
    // ==========
    if (_verbose) std::cout << "Layer #5:\n";
    
    //if (trainFlag) x = torch::relu(batch_norm8(conv8(x)));
    //else x = torch::relu(conv8(x));
    x = torch::relu(conv8(x));
    if (_verbose) std::cout << "\t Conv8: \t" << x.sizes() << std::endl;
    
    x = torch::adaptive_avg_pool2d(x, {1, 1});
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // ==========
    // Layer #6
    // ==========
    if (_verbose) std::cout << "Layer #6:\n";

    //x = torch::relu(convOneXOne1(x));
    //x = torch::leaky_relu(convOneXOne1(x));
    x = convOneXOne1(x);
    if (_verbose) std::cout << "\t 1x1 Conv: \t" << x.sizes() << std::endl;

    // Squeeze
    x = x.squeeze();
    //x = torch::softmax(x.unsqueeze(1), 2);
    
    //if (_numBatch > 1) { // in the case that batch_size is greater than 1
    x = torch::tanh(x.unsqueeze(1));
    //}

    //else { // if batch_size is 1.
    //    x = torch::tanh(x.unsqueeze(0));
    //    x = x.unsqueeze(0);
    //}

    if (_verbose) std::cout << "Last: \n";
    if (_verbose) std::cout << "\t output: \t" << x.sizes() << std::endl;

    return x;
}


/*
void FaceLandmarkNetImpl::infer(torch::Device device, std::string imgPath, std::string modelPath) {
    // Read image
    cv::Mat img = cv::imread(imgPath);
    img.convertTo(img, CV_32FC3); // Convert CV_8UC3 data type to CV_32FC3

    // Resize image
    auto [newW, newH] = _imgRescale;
    cv::Mat resizedImage;
    cv::resize(img, resizedImage, cv::Size2d(newH, newW), 0, 0, cv::INTER_LINEAR);

    // Convert to Tensor
    torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, imgOptions);
    imgTensor = imgTensor.permute({2, 0, 1}); // convert to CxHxW

    torch::load(this, modelPath);
    _output = forward(imgTensor.to(device));

    std::cout << _output.sizes() << std::endl;
}
*/