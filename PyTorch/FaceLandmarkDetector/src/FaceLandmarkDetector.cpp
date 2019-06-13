#include "FaceLandmarkDetector.h"

FaceLandmarkDetector::FaceLandmarkDetector() {
    std::cout << "Constructor" << std::endl;
    
    inputChannel = 3;
    
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannel, 256, 4)
                .stride(1)
                .padding(0)
                .with_bias(false));

    register_module("conv1", conv1);
}