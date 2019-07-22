#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // resize
//#include <torch/torch.h>
#include <torch/data/example.h>
#include <torch/data/transforms/collate.h>
#include <torch/types.h>

#include <filesystem>
//template <typename Target = torch::Tensor>

namespace filesystem = std::experimental::filesystem;

namespace dataWrangling{
class Utilities {
    public:
        static at::Tensor cvImageToTensorConverter(const std::string &imgName, int resizeFactor) {
            cv::Mat imgCV = cv::imread(imgName, CV_LOAD_IMAGE_COLOR);
            
            imgCV.convertTo(imgCV, CV_32FC3);
            cv::resize(imgCV, imgCV, cv::Size2d(resizeFactor, resizeFactor), 0, 0, cv::INTER_LINEAR);

            // Convert to Tensors
            torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
            torch::Tensor imageTensor = torch::from_blob(imgCV.data, {imgCV.rows, imgCV.cols, 3}, imgOptions);
            
            imageTensor = imageTensor.permute({2, 0, 1}); // convert to CxHxW

            return imageTensor.clone();
        }

        static std::vector<std::string> readFileNamesWithAbsPath(std::string folderPath) {

            std::vector<std::string> files;

            for (const auto & entry : filesystem::directory_iterator(folderPath)){
                //std::cout << entry.path() << std::endl;
                files.push_back(entry.path());
            }

            return files;
        }
};

class Resize {
    private:
        int _newWH;

    public:
        Resize(int newWH) : _newWH(newWH){};

        torch::data::Example<> operator() (torch::data::Example<> exampleInput) {
            auto imgTensor = exampleInput.data.toType(torch::kUInt8).clone();
            auto labelTensor = exampleInput.target.clone();

            int origW = (int)imgTensor.size(1);
            int origH = (int)imgTensor.size(2);
            // Convert the image Tensor to cv::Mat with CV_8UC3 data type
            int cvMatSize[2] = {origW, origH};
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

            // Resize cv::Mat
            cv::Mat resizedImage;
            cv::resize(imgCV, resizedImage, cv::Size2d(_newWH, _newWH), 0, 0, cv::INTER_LINEAR);
            resizedImage.convertTo(resizedImage, CV_32FC3);

            // Convert the image to a tensor.
            torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
            torch::Tensor returnTensor = torch::from_blob(resizedImage.data, {resizedImage.rows, resizedImage.cols, 3}, imgOptions);
            returnTensor = returnTensor.permute({2, 0, 1}); // convert to CxHxW

            // Resize label
            for (int coordIdx = 0; coordIdx < 136; ++coordIdx) {
                if (coordIdx %2 == 0) {
                    labelTensor[coordIdx] *= (_newWH / (float)origH);
                } else {
                    labelTensor[coordIdx] *=  (_newWH / (float)origW);
                }
            }
            //labelTensor = labelTensor.toType(torch::kUInt8);

            return {returnTensor.clone(), labelTensor.clone()};
        }
};

class RandomContrastBrightness {
    private:
        float _prob;
        float _alphaFactor;
        float _betaFactor;

    public:
        RandomContrastBrightness(float prob, float contrastFactor, float brightnessFactor) : _prob(prob),  _alphaFactor(contrastFactor), _betaFactor(brightnessFactor) {};

        torch::data::Example<> operator() (torch::data::Example<> exampleInput) {

            if ( rand() % 10 <= _prob*10 ) { // 0.7 probability
                auto inputTensor = exampleInput.data.toType(torch::kUInt8).clone();
                //auto returnTensor = exampleInput.data.clone();
                //auto labelTensor = exampleInput.target.clone();

                cv::RNG rng(cv::getTickCount()); // OpenCV random class
                float alpha = rng.uniform(0.5f, _alphaFactor);
                float beta = rng.uniform(-1*_betaFactor, _betaFactor);

                int origW = (int)inputTensor.size(1);
                int origH = (int)inputTensor.size(2);
                // Convert the image Tensor to cv::Mat with CV_8UC3 data type
                int cvMatSize[2] = {origW, origH};
                cv::Mat imgCVB(2, cvMatSize, CV_8UC1, inputTensor[0].data_ptr());
                cv::Mat imgCVG(2, cvMatSize, CV_8UC1, inputTensor[1].data_ptr());
                cv::Mat imgCVR(2, cvMatSize, CV_8UC1, inputTensor[2].data_ptr());

                // Merge each channel to create colour cv::Mat
                cv::Mat imgCV; // Merged output cv::Mat
                std::vector<cv::Mat> channels;
                channels.push_back(imgCVB);
                channels.push_back(imgCVG);
                channels.push_back(imgCVR);
                cv::merge(channels, imgCV);

                cv::Mat outputCV = cv::Mat::zeros( imgCV.size(), imgCV.type() );
                for( int y = 0; y < imgCV.rows; y++ ) {
                    for( int x = 0; x < imgCV.cols; x++ ) {
                        for( int c = 0; c < imgCV.channels(); c++ ) {
                            outputCV.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>( alpha*imgCV.at<cv::Vec3b>(y,x)[c] + beta );
                        }
                    }
                }

                outputCV.convertTo(outputCV, CV_32FC3);

                // Convert the image to a tensor.
                torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
                torch::Tensor returnTensor = torch::from_blob(outputCV.data, {outputCV.rows, outputCV.cols, 3}, imgOptions);
                returnTensor = returnTensor.permute({2, 0, 1}); // convert to CxHxW

                return {returnTensor.clone(), exampleInput.target.clone()};
            }
            else {
                return {exampleInput.data.clone(), exampleInput.target.clone()};
            }
        }
};

class RandomCrop {
    private:
        float _prob;
        float _moveFactor;

    public:
        RandomCrop(float prob, float moveFactor): _prob(prob), _moveFactor(moveFactor) {};

        torch::data::Example<> operator() (torch::data::Example<> exampleInput) {
            if ( rand() % 10 <= _prob*10 ) { // 0.7 probability
                auto inputTensor = exampleInput.data.toType(torch::kUInt8).clone();
                auto labelTensor = exampleInput.target.clone();

                int origWH = (int)inputTensor.size(1);
                cv::RNG rng(cv::getTickCount()); // OpenCV random class
                float moveX = rng.uniform(-1*_moveFactor, _moveFactor);
                float moveY = rng.uniform(-1*_moveFactor, _moveFactor);

                int cvMatSize[2] = {origWH, origWH};
                cv::Mat imgCVB(2, cvMatSize, CV_8UC1, inputTensor[0].data_ptr());
                cv::Mat imgCVG(2, cvMatSize, CV_8UC1, inputTensor[1].data_ptr());
                cv::Mat imgCVR(2, cvMatSize, CV_8UC1, inputTensor[2].data_ptr());

                // Merge each channel to create colour cv::Mat
                cv::Mat imgCV; // Merged output cv::Mat
                std::vector<cv::Mat> channels;
                channels.push_back(imgCVB);
                channels.push_back(imgCVG);
                channels.push_back(imgCVR);
                cv::merge(channels, imgCV);

                // Resize cv::Mat
                cv::Mat warpGround = (cv::Mat_<float>(2,3) << 1, 0, moveX, 0, 1, moveY);
                cv::Mat outputCV;
                warpAffine(imgCV, outputCV, warpGround,cv::Size(imgCV.rows,imgCV.cols), cv::INTER_LINEAR);// + cv::WARP_INVERSE_MAP);
                outputCV.convertTo(outputCV, CV_32FC3);

                // Convert the image to a tensor.
                torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
                torch::Tensor returnTensor = torch::from_blob(outputCV.data, {outputCV.rows, outputCV.cols, 3}, imgOptions);
                returnTensor = returnTensor.permute({2, 0, 1}); // convert to CxHxW

                for (int coordIdx = 0; coordIdx < 136; ++coordIdx) {
                    if (coordIdx %2 == 0) {
                        labelTensor[coordIdx] += moveX;
                    } else {
                        labelTensor[coordIdx] += moveY;
                    }
                }

                return {returnTensor.clone(), labelTensor.clone()};
            }
            else {
                return {exampleInput.data.clone(), exampleInput.target.clone()};
            }
        }
};


class MiniMaxNormalize {
    //private:
        //int _newWH;

    public:
        //MiniMaxNormalize(int newWH) : _newWH(newWH){};

        torch::data::Example<> operator() (torch::data::Example<> exampleInput) {
            auto imgTensor = exampleInput.data.clone();
            auto labelTensor = exampleInput.target.clone();

            float origW = imgTensor.size(1);
            float origH = imgTensor.size(2);

            imgTensor /= 255;

            // Resize label
            for (int coordIdx = 0; coordIdx < 136; ++coordIdx) {
                if (coordIdx %2 == 0) {
                    labelTensor[coordIdx] /= origH;
                } else {
                    labelTensor[coordIdx] /=  origW;
                }
            }

            return {imgTensor.clone(), labelTensor.clone()};
        }
};
/*//template <typename Target = torch::Tensor>
struct RandomColour : public torch::data::transforms::TensorTransform<torch::Tensor> {
    
    public:

        torch::Tensor operator()(torch::Tensor inputTensor) {
            auto imgTensor = inputTensor.clone();
            //auto labelTensor = exampleInput.target.clone();
            //std::cout << imgTensor.sizes() << std::endl;
            //std::cout << imgTensor[0].sizes() << std::endl;

            if ( rand() % 2 == 1) {
                std::cout << "Random Colour" << std::endl;
                for (int iter=0; iter<3; ++iter) {
                    int rndFactor = rand() % 50;
                    //std::cout << rndFactor << std::endl;
                    //std::cout << imgTensor[iter].sizes() << std::endl;
                    imgTensor[iter] += rndFactor;
                }

                return imgTensor.clone();
            }
        }
};
*/
}; // namespace dataWrangling