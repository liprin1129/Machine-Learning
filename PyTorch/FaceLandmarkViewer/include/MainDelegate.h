#pragma once

#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
//#include "drawLandmarks.hpp"
#include "CameraManager.h"
#include "FaceLandmarksDetector.h"

//using namespace std;
//using namespace cv;
//using namespace cv::face;

class MainDelegate {
	public:
		int mainDelegation(int argc, char** argv);
};