/*
 * ViewManager.cpp
 *
 *  Created on: Jun 7, 2018
 *      Author: user170
 */

#include "ViewManager.hpp"

ViewManager::ViewManager() {
	//cv::namedWindow(LEFT_WINDOW);
	//cvui::init(LEFT_WINDOW);

	this->viewHasLoaded(2, 0.5, 0.5);
}

ViewManager::~ViewManager() {
}

void ViewManager::viewHasLoaded(int argc, ...) {
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
	this->_inCvMat = this->slMatToCvMatConverter(slMat);

	while(this->_key != 'q') {
		if (this->_zed.grab() == sl::SUCCESS) {
			this->_zed.retrieveImage(slMat, sl::VIEW_LEFT, sl::MEM_CPU, this->_width, this->_height);

			this->getFaces(this->_inCvMat);

			//this->_frameCPU = this->createButtonOnWindow(this->_frameCPU, this->_width/15, this->_height/1.2, 150, 50, 0.8);
			//this->mouseHandler("Left");

			//cvui::imshow(LEFT_WINDOW, this->_frameCPU);
			cv::imshow("Left", this->_frameCPU);
			this->_key = cv::waitKey(10);
		}
	}

	cv::destroyWindow("Left");
}
