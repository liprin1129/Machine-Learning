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
	this->_close = false;
	this->_savePeriod = 200; // millisecond

	this->_buttonRandRatioX = 0.45;
	this->_buttonRandRatioY = 0.45;

	this->_score = 0;

	this->_saveButtonFlag = 0;
	// Initialize camera
	// Set input frame size
	// Arguments:
	//		arg1	: width ratio
	//		arg2	: height ratio
	//this->setWidthAndHXeight(0.95, 0.95);

	// Show camera information
	this->printCameraInfo(this->_zed);
}

ViewManager::~ViewManager() {
}

void ViewManager::nameInputView() {
	std::cout << "nameInputView" << std::endl;

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

		if (this->addButton(nameView, width*0.2, height*0.85, 90, 40, "Start") == true) {

			// Try to convert keyboard input to short.
			// If input is empty, nothing happened
			try {
				// Make directory
				//this->makeDir("../../../../Data/Face/Face-SJC/"+employNum+"/");
				this->makeDir("data/Face-SJC/"+employNum+"/");

				// Convert string to short
				boost::lexical_cast<short>(employNum);

				// Go to camera view
				this->cameraView();
				break;
			}
			catch(const boost::bad_lexical_cast &){
			}
		}

		else if (this->addButton(nameView, width*0.6, height*0.85, 90, 40, "Return") == true) {
			//this->_close = false;
			this->_close = true;
			//this->viewHasLoaded(0);
			break;
		}
	}
}

std::string ViewManager::mainView(){
	std::cout << "mainView" << std::endl;

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
			this->_close = true;
			return "STOP";
			break;
		}

		else if (this->addButton(startView, width*0.2, height*0.85, 120, 40, "Enter Employ No.") == true) {
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
	std::cout << "cameraView" << std::endl;

	cv::moveWindow(START_WINDOW, 50, 15);

	// Instance of sl:Mat
	sl::Mat slMat(this->_width, this->_height, sl::MAT_TYPE_8U_C4, sl::MEM_CPU);
	this->_inCvMat = this->slMatToCvMatConverter(slMat);

	while(this->_key != 'q') {
		if (this->_zed.grab() == sl::SUCCESS) {
			// Receive image frame from ZED camera
			this->_zed.retrieveImage(slMat, sl::VIEW_LEFT, sl::MEM_CPU, this->_width, this->_height);
			cv::flip(this->_inCvMat, this->_frameCPU, 1);
			cv::flip(this->_inCvMat, this->flipImg, 1);

			// Start haar cascade face detection
			this->getFaces(this->_frameCPU);

			// Create "Save" button
			this->addButton(this->_frameCPU, airplaneX+15, airplaneY-42, 90, 40, "Save");

			// Create "Score" text
			cvui::printf(this->_frameCPU, this->_width*0.1, this->_height*0.1, 2, 0xff0000, "SCORE: %d", this->_score*100);

			// Create "Return" button
			if (this->addButton(this->_frameCPU, this->_width*0.9, this->_height*0.9, 90, 40, "Return") == true){
				this->_faces.clear();
				//this->viewHasLoaded(0);
				this->_close = true;
				break;
			}

			cvui::imshow(START_WINDOW, this->_frameCPU);
			//cvui::imshow(START_WINDOW, this->_frameCPU);
			this->_key = cv::waitKey(10);
		}
	}
}

void ViewManager::shutterReponse(cv::Mat frame) {
	if (this->_faces.size() != 0) {
		for (int i=0; i < 5; i++) {
			cv::rectangle(frame, cv::Rect(cv::Point(5, 5), cv::Point(this->_width-5, this->_height-5)), cv::Scalar(255, 255, 255), 100);

			//cvui::imshow(START_WINDOW, this->_frameCPU);
			//cv::waitKey(10);
		}
	}
}
void ViewManager::saveFaceLoop() {
	while(!this->_close){
		// Save a face mat
		if (this->_faces.size() != 0) {

			cv::Mat faceCvMat = this->truncateFirstFace(this->flipImg, this->_faces);

			// Save a face mat to a given directory
			this->saveFaceImage(faceCvMat);
		}

		boost::this_thread::sleep(boost::posix_time::millisec(this->_savePeriod));
	}
}

void ViewManager::viewHasLoaded(int argc) {
	std::cout << "viewHasLoaded" << std::endl;

	cv::namedWindow(START_WINDOW);
	cv::moveWindow(START_WINDOW, 700, 288);
	cvui::init(START_WINDOW);

	this->nameInputView();
}
void ViewManager::insertSoccerBall() {
	std::cout << "insertSoccerBall" << std::endl;

	cv::Mat ballImg = cv::imread("data/load-data/airplane.jpg", CV_LOAD_IMAGE_UNCHANGED);
	cv::cvtColor(ballImg, ballImg, cv::COLOR_BGR2RGBA);
	cv::resize(ballImg, ballImg, cv::Size(), 0.1, 0.1, cv::INTER_LINEAR);

	std::cout << "ballImg: " << ballImg.size << std::endl;
	float alpha = 1;
	float beta = 1;
	//cv::addWeighted(this->_frameCPU, alpha, ballImg, 1-alpha, 0.0, this->_frameCPU);

	//int airplaneX = (ballImg.cols*2 + std::rand()/((RAND_MAX + 1u)/(this->_height-ballImg.cols*2)));
	//int airplaneY = (ballImg.rows*2 + std::rand()/((RAND_MAX + 1u)/(this->_width-ballImg.rows*2)));

	std::cout << "insertSoccerBall_airplaneY" << std::endl;
	airplaneY = this->_height*0.1 + std::rand()%(this->_height-ballImg.rows*3);
	airplaneX = this->_width*0.1 + std::rand()%(this->_width-ballImg.cols*3);
	airplaneH = ballImg.rows;
	airplaneW = ballImg.cols;

	while(!this->_close){
		if (this->_faces.size() != 0) {
			/*
			std::cout << airplaneX << ", " << airplaneY << "\t" <<
					airplaneX+airplaneW << ", " << airplaneY+airplaneH << std::endl;
			 */

			ballImg.copyTo(this->_frameCPU(cv::Rect(airplaneX, airplaneY, airplaneW, airplaneH)));

			faceX = this->_faces.front().x;
			faceY = this->_faces.front().y;
			faceW = this->_faces.front().width;
			faceH = this->_faces.front().height;


			if ( ((airplaneX+airplaneW/2) > faceX) & ((airplaneX+airplaneW/2) < faceX+faceW) &
					((airplaneY+airplaneH/2) > faceY) & ((airplaneY+airplaneH/2) < faceY+faceH) ){
				//std::cout << "YES!!!" << std::endl;
				this->shutterReponse(this->_frameCPU);
				airplaneY = this->_height*0.1 + std::rand()%(this->_height-ballImg.rows*3);
				airplaneX = this->_width*0.1 + std::rand()%(this->_width-ballImg.cols*3);

				this->_score += 1;
			}
		}

		std::cout << "insertSoccerBall_thread" << std::endl;
		boost::this_thread::sleep(boost::posix_time::millisec(1));
	}
}
