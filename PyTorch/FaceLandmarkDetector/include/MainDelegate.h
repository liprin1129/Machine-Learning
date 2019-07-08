#ifndef __MAIN_DELEGATE_H__
#define __MAIN_DELEGATE_H__

//#include "Trainer+Inferrer.h"
#include "CustomDataLoader.h"
#include "ImageTansform.h"
#include "FaceLandmarkNet.h"

class MainDelegate {
	public:
		int mainDelegation(int argc, char** argv);

		void train(
			torch::Device device, std::string imgFolderPath, std::string labelCsvFile,
			int numEpoch, int numMiniBatch, int numWorkers,
			float wranglingProb, int resizeFactor, float contrastFactor, float brightnessFactor, float moveFactor);
		void infer(torch::Device device, std::string imgPath, std::string modelPath);
};

#endif /* __MAIN_DELEGATE_H__ */