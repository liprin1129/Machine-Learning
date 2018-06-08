/*
 * ViewManager.cpp
 *
 *  Created on: Jun 7, 2018
 *      Author: user170
 */

#include "ViewManager.hpp"
#define CVUI_IMPLEMENTATION
#include "cvui.h"

ViewManager::ViewManager() {
	cv::namedWindow(START_WINDOW);
	cv::moveWindow(START_WINDOW, 300, 200);
	cvui::init(START_WINDOW);

	this->_buttonRandRatioX = 0.68;
	this->_buttonRandRatioY = 0.9;

	/***********************/
	/*  Initialize camera  */
	/***********************/

	// Set input frame size
	// Arguments:
	//		arg1	: width ratio
	//		arg2	: height ratio
	this->setWidthAndHeight(0.5, 0.5);

	// Show camera information
	this->printCameraInfo(this->_zed);

	/***********************/
	/*  Start View  */
	/***********************/
	//std::cout << this->startView() << std::endl;
	this->viewHasLoaded(0);
}

ViewManager::~ViewManager() {
}

void ViewManager::nameInputView() {
	int width = 500;
	int height = 400;

	cv::Mat nameView = cv::Mat(cv::Size(width, height), CV_8UC3);
	nameView = cv::Scalar(255, 255, 255);

	cvui::text(nameView, width*0.05, height*0.1, "Input your employ number", 1.0, 0x000000);
	std::string employNum;

	while(true) {
		cvui::imshow(START_WINDOW, nameView);
		this->_key = cv::waitKey(10);

		// Check whether keyboard input is numeric or not
		// If input is numeric, append employNum string with the input
		if (isdigit(this->_key)) {
			employNum += this->_key;
			cvui::text(nameView, width*0.4, height*0.3, employNum, 1.0, 0x000000);
		}

		if (this->addButton(nameView, width*0.2, height*0.85, 90, 40, "Enter") == true) {

			// Try to convert keyboard input to short.
			// If input is empty, nothing happened
			try {
				// Make directory
				this->makeDir("../../../../Data/Face/Face-SJC/"+employNum+"/");

				// Convert string to short
				boost::lexical_cast<short>(employNum);
				this->cameraView();
				break;
			}
			catch(const boost::bad_lexical_cast &){
			}
		}

		else if (this->addButton(nameView, width*0.4, height*0.85, 90, 40, "Return") == true) {
			this->viewHasLoaded(0);
			break;
		}
	}
}

std::string ViewManager::mainView(){
	int width = 500;
	int height = 400;

	cv::Mat startView = cv::Mat(cv::Size(width, height), CV_8UC3);
	startView = cv::Scalar(255, 255, 255);

	while(this->_key != 'q') {

		/*
		if (this->addButton(startView, width*0.4, height*0.85, 90, 40, "Start") == true) {
			return "START";
			break;
		}*/
		if (this->addButton(startView, width*0.6, height*0.85, 90, 40, "STOP") == true){
			return "STOP";
			break;
		}

		else if (this->addButton(startView, width*0.2, height*0.85, 90, 40, "Enter name") == true) {
			return "Name";
			break;
		}

		cvui::imshow(START_WINDOW, startView);

		this->_key = cv::waitKey(10);
	}

	return "NO";
}

void ViewManager::nameInputCallback(){

}

bool ViewManager::addButton(cv::Mat frame, int x, int y, std::string msg){
	if (cvui::button(frame, x, y, msg)) {
		return true;
	}

	return false;
}

bool ViewManager::addButton(cv::Mat frame, int x, int y, int width, int height, std::string msg){
	if (cvui::button(frame, x, y, width, height, msg)) {
		return true;
	}

	return false;
}

void ViewManager::cameraView() {
	// Instance of sl:Mat
	sl::Mat slMat(_width, _height, sl::MAT_TYPE_8U_C4, sl::MEM_CPU);
	this->_inCvMat = this->slMatToCvMatConverter(slMat);

	while(this->_key != 'q') {
		if (this->_zed.grab() == sl::SUCCESS) {
			// Receive image frame from ZED camera
			this->_zed.retrieveImage(slMat, sl::VIEW_LEFT, sl::MEM_CPU, this->_width, this->_height);

			// Start haar cascade face detection
			this->getFaces(this->_inCvMat);

			// Create "Save" button
			if (this->addButton(this->_frameCPU, this->_width*this->_buttonRandRatioX, this->_height*this->_buttonRandRatioY, 90, 40, "Save") == true) {

				/*
				this->_tMeter.start();

				while(true){
					this->_tMeter.stop();
					if(this->_tMeter.getTimeMilli() > 10) {
						std::cout << this->_tMeter.getTimeMilli() << "[ms]" << std::endl;
						break;
					}
				}
				*/
				this->_buttonRandRatioX = (1 + std::rand()/((RAND_MAX + 1u)/8))*0.1;
				this->_buttonRandRatioY = (1 + std::rand()/((RAND_MAX + 1u)/8))*0.1;

				if (this->_faces.size() != 0) {
					for (int i=0; i < 5; i++) {
						cv::rectangle(this->_frameCPU, cv::Rect(cv::Point(5, 5), cv::Point(this->_width-5, this->_height-5)), cv::Scalar(255, 255, 255), 30);

						cvui::imshow(START_WINDOW, this->_frameCPU);
						cv::waitKey(10);
					}
				}
			}
			// Create "Return" button
			else if (this->addButton(this->_frameCPU, this->_width*(this->_buttonRandRatioX+0.1), this->_height*this->_buttonRandRatioY, 90, 40, "Return") == true){
				this->viewHasLoaded(0);
				break;
			}

			/*
			// Save a face mat
			if (this->_faces.size() != 0) {
				cv::Mat faceCvMat = this->truncateFirstFace(this->_inCvMat, this->_faces);

				// Save a face mat to a given directory
				this->saveFaceImage(faceCvMat);
			}*/

			cvui::imshow(START_WINDOW, this->_frameCPU);
			this->_key = cv::waitKey(10);
		}
	}
}

void ViewManager::viewHasLoaded(int argc, ...) {
	std::string status = this->mainView();

	if (status == "START") {
		this->cameraView();
	}
	else if (status == "Name"){
		this->nameInputView();
	}
}
