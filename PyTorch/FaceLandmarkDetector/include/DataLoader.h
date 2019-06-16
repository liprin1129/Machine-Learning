#ifndef __IMAGE_DATA_LOADER_H__
#define __IMAGE_DATA_LOADER_H__


#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <vector>
#include <tuple>
#include <list>
#include <regex>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace filesystem = std::experimental::filesystem;

class DataLoader {
    private:
        std::string _rootPath;
        std::vector<std::tuple<std::string, std::string>> _dataset;     // Return dataset vector (image, label) string

        cv::Mat image;
        std::list<float> labels;

    public:
        // Getter
        std::vector<std::tuple<std::string, std::string>> getDataset(){return _dataset;};
        cv::Mat getImage(){return image;};
        std::list<float> getLabels(){return labels;};

        DataLoader(std::string path);                                               // Constructor, string arguments
        void readDataAndLabels();                                                   // recursively read files from root folder
        void labelStr2Float(std::tuple<std::string, std::string> filePath, bool norm = true); // Convert string of label to float
        
        cv::Mat readImage2CVMat(std::string filePath, bool norm=true);              // Read an image data into cv::Mat
        std::tuple<float, float> labelNormalizer(int col, int row, float X, float Y);                      // MinMax normalize for label data
};

#endif // __IMAGE_DATA_LOADER_H__s