/*
 * HighGuiHandler.hpp
 *
 *  Created on: Jun 6, 2018
 *      Author: user170
 */

#ifndef HIGHGUIHANDLER_HPP_
#define HIGHGUIHANDLER_HPP_

#include "CommonHeaders.hpp"

class HighGuiHandler {
public:
	HighGuiHandler();
	virtual ~HighGuiHandler();

	cv::Mat createButtonOnWindow(cv::Mat img, int x, int y, int width, int height, double alpha);
};

#endif /* HIGHGUIHANDLER_HPP_ */
