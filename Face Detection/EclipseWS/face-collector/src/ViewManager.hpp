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
	void viewHasLoaded(int argc, ...);
};

#endif /* VIEWMANAGER_HPP_ */
