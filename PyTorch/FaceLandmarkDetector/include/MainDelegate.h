#ifndef __MAIN_DELEGATE_H__
#define __MAIN_DELEGATE_H__

#include <math.h> 
//#include "Trainer+Inferrer.h"
#include "CustomDataLoader.h"
#include "ImageTansform.h"
#include "FaceLandmarkNet.h"

class MainDelegate {
	public:
		int mainDelegation(int argc, char** argv);

		void train
		(
			bool verbose, torch::Device device, std::string imgFolderPath, std::string labelCsvFile,
			float learningRate, int numEpoch, int numMiniBatch, int numWorkers,
			float wranglingProb, int resizeFactor, float contrastFactor, float brightnessFactor, float moveFactor
		);
		
		void infer(torch::Device device, std::string imgPath, std::string modelPath);
};

#endif /* __MAIN_DELEGATE_H__ */