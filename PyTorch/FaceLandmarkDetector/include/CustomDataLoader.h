#ifndef __IMAGE_DATA_LOADER_H__
#define __IMAGE_DATA_LOADER_H__

#include <torch/torch.h>

#include <iostream>
#include <fstream>
#include <string>
#include <experimental/filesystem>

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // resize

#include "ImageTansform.h"

namespace filesystem = std::experimental::filesystem;

class CustomDataset: public torch::data::Dataset<CustomDataset> {
    private:
        //torch::Tensor _states, _labels; // Return Tensors
        std::vector<std::tuple<std::string, std::vector<float>>> _dataset;     // Return dataset vector (image, label) string
        std::string _locCSV;
        std::string _locImages;

        std::tuple<int, int> _rescale;

        bool _verbose;
        //std::unique_ptr<Rescale> _rescale;
        void checkcvMatNan(cv::Mat img, std::string dTypeStr);

    public:
        //GETTER
        std::tuple<int, int> getRescale(){return _rescale;};

        explicit CustomDataset(const std::string& locCSV, const std::string& locImages, std::tuple<int, int> newSize, bool verbose);

        torch::data::Example<> get(size_t index) override;
        torch::Tensor read_data(const std::string &loc);

        void readCSV(const std::string &loc);

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {
            //std::cout << _dataset.size() << std::endl;
            return _dataset.size();
        };
};

#endif // __IMAGE_DATA_LOADER_H__