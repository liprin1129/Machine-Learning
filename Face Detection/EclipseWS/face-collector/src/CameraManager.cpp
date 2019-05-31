/*
 * CameraManager.cpp
 *
 *  Created on: May 30, 2018
 *      Author: user170
 */

#include "CameraManager.hpp"

//#include <opencv2/objdetect.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/cudaobjdetect.hpp>

CameraManager::CameraManager() {
	// Initialize _key to empty space
	this->_key = ' ';

	this->parameterInitializer(sl::RESOLUTION_HD1080, sl::DEPTH_MODE_PERFORMANCE, sl::UNIT_MILLIMETER);
	this->openCamera();

	//this->cameraManagerHasLoaded(2, 0.5, 0.5);

	//std::cout << "WIDTH: " << _width << std::endl;
}

CameraManager::~CameraManager(){
	std::cout << "CameraManager Deconstructor: Camera close" << std::endl;
	this->_zed.close();
}

void CameraManager::parameterInitializer(sl::RESOLUTION res) {
	this->_initParams.camera_resolution = res;
}

void CameraManager::parameterInitializer(sl::RESOLUTION res, sl::DEPTH_MODE depth, sl::UNIT unit) {
	this->_initParams.camera_resolution = res; //sl::RESOLUTION_HD1080;
	this->_initParams.depth_mode = depth; //sl::DEPTH_MODE_PERFORMANCE;
	this->_initParams.coordinate_units = unit; //sl::UNIT_MILLIMETER;
}

// Open camera.
// Return:
//		1: if openCamera() occurs error.
int CameraManager::openCamera() {
	std::cout << "openCamera" << std::endl;
	sl::ERROR_CODE err = this->_zed.open(this->_initParams);

	if (err != sl::SUCCESS) {
		std::cout << sl::toString(err).c_str() << std::endl;
		this->_zed.close();
		return 1; // Quit if an error occurred
	}

	return 0;
}

// Show camera information
// Information:
//		camera model
//		serial number
//		firmware version
//		resolution
//		fps
void CameraManager::printCameraInfo(sl::Camera &zedCameraObject){
	std::cout << "ZED Model             : " <<
			sl::toString(zedCameraObject.getCameraInformation().camera_model).c_str() << std::endl;
	std::cout << "ZED Serial Number     : " <<
			zedCameraObject.getCameraInformation().serial_number << std::endl;
	std::cout << "ZED Serial Firmware   : " <<
			zedCameraObject.getCameraInformation().firmware_version << std::endl;
	std::cout << "ZED Camera Resolution : (" <<
			(int)zedCameraObject.getResolution().width << ", " << (int)zedCameraObject.getResolution().height << ")" << std::endl;
	std::cout << "ZED FPS Resolution    : ( " << this->_width << ", " << this->_height << ")" << std::endl;
	std::cout << "ZED Serial FPS        : " <<
			(int)zedCameraObject.getCameraFPS() << std::endl;
}

// Convert sl::Mat to cv::Mat
cv::Mat CameraManager::slMatToCvMatConverter(sl::Mat &slMat) {
	std::cout << "GPU SL Mat" << std::endl;

	if (slMat.getMemoryType() == sl::MEM_GPU) {
		std::cout << "GPU SL Mat" << std::endl;
		slMat.updateCPUfromGPU();
	}

	int cvType = -1;
	switch (slMat.getDataType()){
		case sl::MAT_TYPE_32F_C1: cvType = CV_32FC1; break;
		case sl::MAT_TYPE_32F_C2: cvType = CV_32FC2; break;
		case sl::MAT_TYPE_32F_C3: cvType = CV_32FC3; break;
		case sl::MAT_TYPE_32F_C4: cvType = CV_32FC4; break;
		case sl::MAT_TYPE_8U_C1: cvType = CV_8UC1; break;
		case sl::MAT_TYPE_8U_C2: cvType = CV_8UC2; break;
		case sl::MAT_TYPE_8U_C3: cvType = CV_8UC3; break;
		case sl::MAT_TYPE_8U_C4: cvType = CV_8UC4; break;
		default: break;
	}

	/*
	return cv::Mat(
			slMat.getHeight(),
			slMat.getWidth(),
			cvType,
			slMat.getPtr<sl::uchar1>(memType)
			);
	*/
	return cv::Mat((int) slMat.getHeight(),
			(int) slMat.getWidth(),
			cvType,
			slMat.getPtr<sl::uchar1>(sl::MEM_CPU),
			slMat.getStepBytes(sl::MEM_CPU));
}

void CameraManager::setWidthAndHeight(float _widthRatio, float _heightRatio){
	std::cout << "setWidthAndHeight" << std::endl;
	sl::Resolution resolution = this->_zed.getResolution();
	this->_width = resolution.width * _widthRatio;
	this->_height= resolution.height * _heightRatio;
}

cv::Mat CameraManager::getOneCvMat(){
	std::cout << "getOneCvMat" << std::endl;
	this->_zed.grab();
	this->_zed.retrieveImage(this->_inSlMat, sl::VIEW_LEFT, sl::MEM_CPU, this->_width, this->_height);

	return this->_inCvMat;
}
/*
int CameraManager::cameraManagerHasLoaded(int argc, ...) {

	va_list argv;
	va_start(argv, argc);

	// Set input frame size
	// Arguments:
	//		arg1	: width ratio
	//		arg2	: height ratio
	this->setWidthAndHeight(va_arg(argv, double), va_arg(argv, double));

	// Show camera information
	this->printCameraInfo(this->_zed);


	// Instance of sl:Mat
	sl::Mat slMat(_width, _height, sl::MAT_TYPE_8U_C4, sl::MEM_CPU);
	this->_inSlMat = slMat;

	// Wrapper of sl::Mat to cv::Mat
	this->_inCvMat = this->slMatToCvMatConverter(slMat);

	cv::namedWindow("Left", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("Left", 30, 30);

	while(this->_key != 'q') {
		if (this->_zed.grab() == sl::SUCCESS) {
			this->_zed.retrieveImage(slMat, sl::VIEW_LEFT, sl::MEM_CPU, this->_width, this->_height);

			this->getFaces(this->_inCvMat);

			//this->_frameCPU = this->createButtonOnWindow(this->_frameCPU, this->_width/15, this->_height/1.2, 150, 50, 0.8);
			//this->mouseHandler("Left");

			cv::imshow("Left", this->_frameCPU);
			this->_key = cv::waitKey(10);
		}
	}

	cv::destroyWindow("Left");

	return 0;
}
*/
