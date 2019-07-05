#include <iostream>
#include "FaceLandmarkNet.h"
#include "DataLoader.h"

class TrainerInferrer {
    private:
        FaceLandmarkNet fln{nullptr};

        std::tuple<int, int> _imgRescale; // new size
        bool _verbose;

        int _numEpoch;
        int _numBatch;

    public:
        TrainerInferrer(int numEpoch, int numBatch, std::tuple<int, int> imgRescale, bool verbose);

        void train(torch::Device device, std::string imgFolderPath, std::string labelCsvFile);
        void infer(torch::Device device, std::string imgPath, std::string modelPath);
        void checkTensorImgAndLandmarks(torch::Tensor const &imgTensor, torch::Tensor const &inferredTensor, int newWH);
        void checkTensorImgAndLandmarks(int epoch, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor, torch::Tensor const &gtLabelTensor, int newWH);
};