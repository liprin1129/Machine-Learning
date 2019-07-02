#ifndef __FACE_LANDMARK_NET_H__
#define __FACE_LANDMARK_NET_H__

#include <torch/torch.h>
#include <iostream>
#include <DataLoader.h>

class FaceLandmarkNetImpl : public torch::nn::Module {
    private:
        int64_t inputChannel;
        bool _verbose;
        bool _testFlag;

        int _numEpoch;
        int _numBatch;
        std::tuple<int, int> _imgRescale;

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr},
                            conv5{nullptr}, conv6{nullptr}, conv7{nullptr}, conv8{nullptr};

        torch::nn::Conv2d convOneXOne1{nullptr};

        torch::nn::BatchNorm batch_norm1{nullptr}, batch_norm2{nullptr}, batch_norm3{nullptr}, batch_norm4{nullptr},
                                batch_norm5{nullptr}, batch_norm6{nullptr}, batch_norm7{nullptr}, batch_norm8{nullptr};

    public:
        // GETTER
        torch::nn::Conv2d getConv(){return conv1;};

        FaceLandmarkNetImpl(int numEpoch, int numBatch, std::tuple<int, int> imgRescale, bool verbose, bool testFlag);
        torch::Tensor forward(torch::Tensor x, bool trainFlag);

        void checkTensorImgAndLandmarks(int epoch, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor, torch::Tensor const &gtLabelTensor);

        at::Tensor cvMat2Tensor(cv::Mat cvMat, torch::Device device);
        at::Tensor floatList2Tensor(std::list<float> floatList, torch::Device device);

        void train(torch::Device device, torch::optim::Optimizer &optimizer);
        void showTrainInfo(cv::Mat cvImg, std::list<float> listLabel, at::Tensor &inX, at::Tensor &label);
        void outputImage(cv::Mat cvImg, at::Tensor output, int epoch);
        void outputImage(cv::Mat cvImg, std::list<float> output);
        //void outputImage(DataLoader &dl, torch::Device device, int epoch);

};
TORCH_MODULE(FaceLandmarkNet);

#endif //__FACE_LANDMARK_NET_H__