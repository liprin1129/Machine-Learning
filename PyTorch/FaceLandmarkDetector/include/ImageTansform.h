#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // resize
//#include <torch/torch.h>
#include <torch/data/example.h>
#include <torch/data/transforms/collate.h>
#include <torch/types.h>

//template <typename Target = torch::Tensor>
template <typename Target>
struct MyResize : public torch::data::transforms::TensorTransform<Target> {
    private:
        int _newWH;
    
    public:
        MyResize(int newWH)
            : _newWH(newWH) {}

        torch::Tensor operator()(torch::Tensor inputTensor) {
            torch::Tensor imgTensor = inputTensor.toType(torch::kUInt8).clone();
            //std::cout << imgTensor << std::endl;
            //std::cout << imgTensor.max() << ", " << imgTensor.min() << std::endl;

            // Convert the image Tensor to cv::Mat with CV_8UC3 data type
            int cvMatSize[2] = {(int)imgTensor.size(1), (int)imgTensor.size(2)};
            cv::Mat imgCVB(2, cvMatSize, CV_8UC1, imgTensor[0].data_ptr());
            cv::Mat imgCVG(2, cvMatSize, CV_8UC1, imgTensor[1].data_ptr());
            cv::Mat imgCVR(2, cvMatSize, CV_8UC1, imgTensor[2].data_ptr());

            // Merge each channel to create colour cv::Mat
            cv::Mat imgCV; // Merged output cv::Mat
            std::vector<cv::Mat> channels;
            channels.push_back(imgCVB);
            channels.push_back(imgCVG);
            channels.push_back(imgCVR);
            cv::merge(channels, imgCV);

            cv::Mat resizedImage;
            cv::resize(imgCV, resizedImage, cv::Size2d(_newWH, _newWH), 0, 0, cv::INTER_LINEAR);
            //std::cout << resizedImage << std::endl;

            // Convert the image.
            resizedImage.convertTo(resizedImage, CV_32FC3);
            torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
            torch::Tensor returnTensor = torch::from_blob(resizedImage.data, {resizedImage.rows, resizedImage.cols, 3}, imgOptions);
            returnTensor = returnTensor.permute({2, 0, 1}); // convert to CxHxW
            //std::cout << returnTensor << std::endl;

            return returnTensor;
        }
};

