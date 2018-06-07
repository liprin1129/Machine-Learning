/*
 * ViewManager.hpp
 *
 *  Created on: Jun 7, 2018
 *      Author: user170
 */

#ifndef VIEWMANAGER_HPP_
#define VIEWMANAGER_HPP_

#include "cvui.h"
#include "CommonHeaders.hpp"

#include "ImageManager.hpp"
#include "CameraManager.hpp"

#define LEFT_WINDOW "Left"

class ViewManager: public CameraManager, public ImageManager {

public:
	ViewManager();
	virtual ~ViewManager();

	void viewHasLoaded(int argc, ...);
};

#endif /* VIEWMANAGER_HPP_ */
