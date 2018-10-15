/*
 * ViewManager.hpp
 *
 *  Created on: Jun 7, 2018
 *      Author: user170
 */

#ifndef VIEWMANAGER_HPP_
#define VIEWMANAGER_HPP_

#include "CommonHeaders.hpp"

#include "ImageManager.hpp"
#include "CameraManager.hpp"
#include "FileSystemManager.hpp"

#define LEFT_WINDOW "Left"
#define START_WINDOW "Start"

class ViewManager: public CameraManager, public ImageManager, public FileSystemManager {
private:
	float _buttonRandRatioX;
	float _buttonRandRatioY;
	bool _close;
	int _savePeriod;
	int _saveButtonFlag;

	cv::Mat flipImg;

	int airplaneY, airplaneX, airplaneH, airplaneW;

	int faceX, faceY, faceW, faceH;

	int _score;

public:
	ViewManager();
	virtual ~ViewManager();

	// Initial view
	std::string mainView();

	// Name input view
	void nameInputView();
	// Receive keyboard input in the name input view
	void nameInputCallback();

	// Create Button
	bool addButton(cv::Mat frame, int x, int y, std::string msg);
	bool addButton(cv::Mat frame, int x, int y, int width, int height, std::string msg);

	void cameraView();

	// Call this function in a loop to save detected face
	void saveFaceLoop();

	// Shutter animation which shows white boundary
	void shutterReponse(cv::Mat frame);

	void insertSoccerBall();
	void viewHasLoaded(int argc);
};

#endif /* VIEWMANAGER_HPP_ */
