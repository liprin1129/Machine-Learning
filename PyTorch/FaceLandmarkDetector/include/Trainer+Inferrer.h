#pragma once

#include <iostream>
#include "FaceLandmarkNet.h"
#include "CustomDataLoader.h"

class TrainerInferrer {

    public:

        void testShow(int count, int resizeFactor, const torch::Tensor &imgTensor, const torch::Tensor &labelTensor);
        void testSave(int count, int rescaleFactor, const torch::Tensor &imgTensor, const torch::Tensor &labelTensor, char* outputName);
        void writeCSV(int count, torch::Tensor const &labelTensor);

		void train
		(
            FaceLandmarkNet fln,
			bool verbose, torch::Device device, std::string imgFolderPath, std::string labelCsvFile,
			float learningRate, int numEpoch, int numMiniBatch, int numWorkers,
			float wranglingProb, int resizeFactor, float contrastFactor, float brightnessFactor, float moveFactor,
			int saveInterval
		);

        void inferStereo(
            FaceLandmarkNet fln, 
            //cv::Mat leftImg, cv::Mat rightImg, 
            const at::Tensor &leftImageTensor, const at::Tensor &rightImageTensor, 
            torch::Device device
        );

        void inferMono(FaceLandmarkNet fln, const at::Tensor &imageTensor, torch::Device device, int imgCount);
};