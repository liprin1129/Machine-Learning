/*
 * HighGuiHandler.cpp
 *
 *  Created on: Jun 6, 2018
 *      Author: user170
 */

#include "HighGuiHandler.hpp"

static void mouse_callback(int  event, int  x, int  y, int  flag, void *param)
{
	if (event == cv::EVENT_MOUSEMOVE) {
		std::cout << "(" << x << ", " << y << ")" << std::endl;
	}
}

HighGuiHandler::HighGuiHandler() {
	/*
    cv::Mat3b img(200, 200, cv::Vec3b(0, 255, 0));

    cv::namedWindow("example");
    cv::setMouseCallback("example", mouse_callback);

    cv::imshow("example", img);
    cv::waitKey();
    */
}

HighGuiHandler::~HighGuiHandler() {

}

cv::Mat HighGuiHandler::createButtonOnWindow(
		cv::Mat frame,
		int x,
		int y,
		int width,
		int height,
		double alpha){

	cv::Mat outputImg;
	cv::Mat overlayImg;

	frame.copyTo(outputImg);
	frame.copyTo(overlayImg);

	cv::rectangle(overlayImg, cv::Rect(x, y, width, height), cv::Scalar(255, 255, 255), -1);
	cv::addWeighted(overlayImg, alpha, outputImg, 1 - alpha, 0, outputImg);

	cv::putText(outputImg, "Capture", cv::Point(x+15, y+30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 3, 8);
	//cv::rectangle(overayImg, cv::Point(x, y), cv::Point(x+width, y+height), cv::Scalar(0, 255, 0), -1);

	//cv::addWeighted(overayImg, alpha, img, 1-alpha, 0, outputImg);

	return outputImg;
}

void HighGuiHandler::mouseHandler(std::string windowName) {
	cv::setMouseCallback(windowName, mouse_callback);
}
