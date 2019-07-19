#include "MainDelegate.h"

/*

To run this code:

	- when infering:
./FaceLandmarkDetector trained-model image1 image2
./FaceLandmarkDetector ./checkpoints/Trained-models/backup-models/model-2000.pt /DATASETs/Face/Landmarks/300W-Dataset/300W/Data/indoor_001.png /DEVs/Machine-Learning/PyTorch/FaceLandmarkDetector/TestImages/Endo.png

*/

int main(int argc, char** argv) {
	MainDelegate delegation;

	return delegation.mainDelegation(argc, argv);
}