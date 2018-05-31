/*
 * ImageManager.cpp
 *
 *  Created on: May 31, 2018
 *      Author: user170
 */

#include "ImageManager.hpp"

ImageManager::ImageManager() {

	this->imageManagerHasLoaded(1, "../data/haarcascade_frontalface_alt_gpu.xml");
}

ImageManager::~ImageManager() {
	std::cout << "Bye Bye" <<std::endl;
}

int ImageManager::imageManagerHasLoaded(int argc, ...) {
	// Read multiple arguments
	va_list argv;
	va_start(argv, argc);

	// Print out gpu device information
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	// Assign cascade file name
	this->_cascadeFileName = va_arg(argv, char*);
	std::cout << this->_cascadeFileName << std::endl;

	// Instantiate cv::cuda::CascadeClassifier
	this->_cascade_gpu = cv::cuda::CascadeClassifier::create(this->_cascadeFileName);

	// Camera setting
	this->_cm.parameterInitializer(sl::RESOLUTION_HD1080);
	this->_cm.openCamera();
	this->_cm.setWidthAndHeight(0.5, 0.5);

	sl::Mat slMat(this->_cm._width , this->_cm._height, sl::MAT_TYPE_8U_C4, sl::MEM_CPU);
	cv::Mat cvMat = this->_cm.slMatToCvMatConverter(slMat);

	// Loop until 'q' is pressed
	char key = ' ';
	while (key != 'q') {

		if (this->_cm._zed.grab() == sl::SUCCESS) {
			this->_cm._zed.retrieveImage(slMat, sl::VIEW_LEFT, sl::MEM_CPU, this->_cm._width, this->_cm._height);

			cv::imshow("Image-Left", cvMat);

			// Handle key event
			key = cv::waitKey(10);
			//processKeyEvent(zed, key); // @suppress("Invalid arguments")
		}
	}
	return 0;
}
