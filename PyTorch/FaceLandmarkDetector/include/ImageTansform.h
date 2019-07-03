#ifndef __IMAGE_TRANSFORM_H__
#define __IMAGE_TRANSFORM_H__

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // resize
//k#include <typeinfo>
//#include <tuple>
//#include <vector>

class Rescale {
    private:
        int _outputSizeInt;
        std::tuple<int, int> _outputSizeTuple;
        std::tuple<cv::Mat, std::vector<float>> _resizedDataCVandFloat;

        int _newW, _newH;

        const std::type_info &_typeId;
    
    public:
        // GETTERs
        std::tuple<cv::Mat, std::vector<float>> getResizedDataCVandFloat(){return _resizedDataCVandFloat;};

        // METHODs
        Rescale(int outputSize): _outputSizeInt(outputSize), _typeId(typeid(int)) {
            if (typeid(outputSize) != typeid(int)){
                std::fprintf(stderr, "Rescale ERROR: data type of output size is incorrect.\n");
                exit(-1);
            }

            //std::fprintf(stdout, "outputSizeInt: %d\n", _outputSizeInt);
        };

        Rescale(std::tuple<int, int> outputSize): _outputSizeTuple(outputSize), _typeId(typeid(std::tuple<int, int>)) {
            if (typeid(outputSize) != typeid(std::tuple<int, int>)) {
                std::fprintf(stderr, "Rescale ERROR: data type of output size is incorrect.\n");
                exit(-1);
            }

            auto [A, B] = _outputSizeTuple;
            //std::fprintf(stdout, "outputSizeTuple: %d, %d\n", A, B);
        };

        Rescale operator() (cv::Mat const& image, std::vector<int> const &landmarks) {
            int w = image.rows;
            int h = image.cols;
            
            //std::cout << image.size << std::endl;
            //std::fprintf(stdout, "W: %d, H: %d\n", w, h);

            if (_typeId == typeid(int) and _outputSizeInt != 0) {
                if (h > w) {
                    //std::cout << "W<H" << std::endl;
                    //_newH =  (int)_outputSizeInt * h/(float)w;
                    _newH =  _outputSizeInt * h/w;
                    _newW = _outputSizeInt;
                }
                else {
                    //std::cout << "W>H" << std::endl;
                    _newH = _outputSizeInt;
                    //_newW =  (int)_outputSizeInt * w/(float)h;
                    _newW =  _outputSizeInt * w/h;
                }
            }
            else {
                _newH = h;
                _newW = w;
            }
            
            if(std::get<0>(_outputSizeTuple) != 0 and std::get<1>(_outputSizeTuple) != 0){
                auto [newW, newH] = _outputSizeTuple;
                _newW = newW;
                _newH = newH;
            }
            else {
                _newW = w;
                _newH = h;
            }
            
            cv::Mat resizedImage;
            std::vector<float> resizedLabel;

            // Rescale the image
            cv::resize(image, resizedImage, cv::Size2d(_newH, _newW), 0, 0, cv::INTER_LINEAR);

            // Rescale the labels
            int coord = 0;
            for (auto landmark: landmarks) {
                if (++coord % 2 == 1) resizedLabel.push_back(landmark * _newH/h);
                else resizedLabel.push_back(landmark * _newW/w);
            }
            //std::fprintf(stdout, "newW: %d, newH: %d\n", _newW, _newH);

            //std::cout << resizedImage.size << std::endl;

            //for (auto landmark: resizedLabel) {
                //std::cout << landmark << ", ";
            //}
            //std::cout << std::endl;

            _resizedDataCVandFloat = std::make_tuple(resizedImage, resizedLabel); // data to be returned

            return *this;
        }


        Rescale operator() (torch::Tensor const &imgTensor, torch::Tensor const &labelTensor) {
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

            int w = imgCV.rows;
            int h = imgCV.cols;

            if (_typeId == typeid(int)) {
                if (h > w) {
                    //std::cout << "W<H" << std::endl;
                    //_newH =  (int)_outputSizeInt * h/(float)w;
                    _newH =  _outputSizeInt * h/w;
                    _newW = _outputSizeInt;
                }
                else {
                    //std::cout << "W>H" << std::endl;
                    _newH = _outputSizeInt;
                    //_newW =  (int)_outputSizeInt * w/(float)h;
                    _newW =  _outputSizeInt * w/h;
                }
            }
            else {
                auto [newW, newH] = _outputSizeTuple;
                _newW = newW;
                _newH = newH;
            }

            cv::Mat resizedImage;
            std::vector<float> resizedLabel;

            // Rescale the image
            cv::resize(imgCV, resizedImage, cv::Size2d(_newH, _newW), 0, 0, cv::INTER_LINEAR);

            // Convert the label Tensor to vector
            std::vector<float> resizedLabelVector;

            for (int i=0; i<labelTensor.size(1); ++i) {
                resizedLabelVector.push_back(labelTensor[0][i].item<float>());
            }

            // Rescale the labels
            int coord = 0;
            for (auto landmark: resizedLabelVector) {
                if (++coord % 2 == 1) resizedLabel.push_back(landmark * _newH/h);
                else resizedLabel.push_back(landmark * _newW/w);
            }

            _resizedDataCVandFloat = std::make_tuple(resizedImage, resizedLabel); // data to be returned
            //_resizedDataTensor = {};
            return *this;
        }
};

#endif // __IMAGE_TRANSFORM_H__