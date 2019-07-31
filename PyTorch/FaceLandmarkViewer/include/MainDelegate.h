#pragma once

#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
//#include "drawLandmarks.hpp"
#include "CameraManager.h"
#include "FaceLandmarksDetector.h"
#include "DepthEstimator.h"

class MainDelegate {
	public:
		int mainDelegation(int argc, char** argv);
};