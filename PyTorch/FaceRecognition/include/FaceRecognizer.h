#pragma once

#include <iostream>
#include <experimental/filesystem>
//#include <vector>
//#include <algorithm>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // resize

#include "ImageTansform.h"
#include "FaceLandmarkNet.h"

namespace filesystem = std::experimental::filesystem;

class FaceRecognizer {
    private:
        std::vector<std::vector<float>> meanAndVar;
        
    public:
        std::vector<std::string> csvFinder(const std::string &rootDir);
        std::vector<std::vector<float>> csvReader(const std::string &filePath);
        
        void faceRecognition(FaceLandmarkNet fln, const at::Tensor &imageTensor, const std::vector<std::vector<float>> &meanAndVarVec, torch::Device device);
};
