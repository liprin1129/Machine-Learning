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

public:
	ViewManager();
	virtual ~ViewManager();

	std::string mainView();
	void nameInputView();

	void nameInputCallback();
	// Create Button
	bool addButton(cv::Mat frame, int x, int y, std::string msg);
	bool addButton(cv::Mat frame, int x, int y, int width, int height, std::string msg);

	void cameraView();

	// Call this function in a loop to save detected face
	void saveFaceLoop();

	void viewHasLoaded(int argc);

	void threadTest1();
	void threadTest2(const std::string tt);
};

#endif /* VIEWMANAGER_HPP_ */
