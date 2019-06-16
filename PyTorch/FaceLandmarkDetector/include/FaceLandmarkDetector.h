#ifndef __FACE_LANDMARK_DETECTOR_H__
#define __FACE_LANDMARK_DETECTOR_H__

#include <torch/torch.h>
#include <iostream>

class FaceLandmarkDetector : public torch::nn::Module {
    private:
        int64_t inputChannel;
        bool _verbose;
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr},
                            conv5{nullptr}, conv6{nullptr}, conv7{nullptr}, conv8{nullptr};

        torch::nn::Conv2d convOneXOne1{nullptr}, convOneXOne2{nullptr};

        torch::nn::BatchNorm batch_norm1{nullptr}, batch_norm2{nullptr}, batch_norm3{nullptr}, batch_norm4{nullptr},
                                batch_norm5{nullptr}, batch_norm6{nullptr}, batch_norm7{nullptr}, batch_norm8{nullptr};

    public:
        // GETTER
        torch::nn::Conv2d getConv(){return conv1;};

        FaceLandmarkDetector(bool verbose);

        torch::Tensor forward(torch::Tensor x);
};

#endif //__FACE_LANDMARK_DETECTOR_H__