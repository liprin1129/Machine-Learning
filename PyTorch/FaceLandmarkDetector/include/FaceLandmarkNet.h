#ifndef __FACE_LANDMARK_NET_H__
#define __FACE_LANDMARK_NET_H__

#include <torch/torch.h>
#include <iostream>
#include <CustomDataLoader.h>

class FaceLandmarkNetImpl : public torch::nn::Module {
    private:
        bool _verbose;
        bool _mode;
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr},
                            conv5{nullptr}, conv6{nullptr}, conv7{nullptr}, conv8{nullptr},
                            conv9{nullptr}, conv10{nullptr};

        //torch::nn::Conv2d convOneXOne1{nullptr};

        torch::nn::BatchNorm batch_norm1{nullptr}, batch_norm2{nullptr}, batch_norm3{nullptr}, batch_norm4{nullptr},
                                batch_norm5{nullptr}, batch_norm6{nullptr}, batch_norm7{nullptr}, batch_norm8{nullptr},
                                batch_norm9{nullptr}, batch_norm10{nullptr};


    public:
        FaceLandmarkNetImpl(int inputChannel, bool verbose); // mode [0: infering, 1: training]
        
        torch::Tensor forward(torch::Tensor x);
        
        //void train(torch::Device device, torch::optim::Optimizer &optimizer);
        //void infer(torch::Device device, std::string imgPath, std::string modelPath);
        //void checkTensorImgAndLandmarks(int epoch, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor, torch::Tensor const &gtLabelTensor);
};
TORCH_MODULE(FaceLandmarkNet);

#endif //__FACE_LANDMARK_NET_H__