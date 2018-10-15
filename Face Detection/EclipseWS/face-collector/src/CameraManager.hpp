/*
 * CameraManager.hpp
 *
 *  Created on: May 30, 2018
 *      Author: user170
 */

#ifndef CAMERAMANAGER_HPP_
#define CAMERAMANAGER_HPP_

#include "CommonHeaders.hpp"
//#include "ImageManager.hpp"
//#include "ViewManager.hpp"

class CameraManager {

protected:
	char _key;
	// Create a ZED camera object
	//sl::Camera _zed;

	// Create configuration parameter object
	//sl::InitParameters _initParams;

	// Image output width and height
	int _height;
	int _width;

	//sl::Mat _inSlMat;
	sl::Mat _inSlMat;
	cv::Mat _inCvMat;
	//cv::Mat _overlapMat;

public:
	CameraManager();
	~CameraManager();

	int height() const {return _height;}
	int width() const {return _width;}
	//sl::Mat inSlMat() const {return _inSlMat;}

	// Call this function after camera open
	// Arguments:
	//		_widthRatio		: 0 ~ 1
	//		_heightRatio	: 0 ~ 1
	void setWidthAndHeight(float _widthRatio, float _heightRatio);

	// Create a ZED camera object
	sl::Camera _zed;

	// Create configuration parameter object
	sl::InitParameters _initParams;

	void parameterInitializer(sl::RESOLUTION res);
	void parameterInitializer(sl::RESOLUTION res, sl::DEPTH_MODE depth, sl::UNIT unit);

	// Return 1 if openCamera() occurs error
	int openCamera();

	// Show camera information
	void printCameraInfo(sl::Camera &zedCameraObject);

	// sl::Mat to cv::Mat converter
	cv::Mat slMatToCvMatConverter(sl::Mat &slMat);

	// Convert sl::Mat to cv::Mat
	void linkSlMatToCvMat(int width, int height, sl::MAT_TYPE _pixelType);

	// Get one sl::Mat from camera
	cv::Mat getOneCvMat();

	int cameraManagerHasLoaded(int argc, ...);
};

#endif /* CAMERAMANAGER_HPP_ */
