/*
 * CameraManager.cpp
 *
 *  Created on: May 30, 2018
 *      Author: user170
 */

#include "CameraManager.hpp"

//#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>

/*CameraManager::CameraManager(int width, int height, sl::MAT_TYPE pixelType) {
	this->_width = width;
	this->_height = height;
}*/

void CameraManager::parameterInitializer() {
	this->_initParams.camera_resolution = sl::RESOLUTION_HD1080;
	this->_initParams.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
	this->_initParams.coordinate_units = sl::UNIT_MILLIMETER;
}

// Open camera.
// Return:
//		1: if openCamera() occurs error.
int CameraManager::openCamera() {
	sl::ERROR_CODE err = this->_zed.open(this->_initParams);

	if (err != sl::SUCCESS) {
		std::cout << sl::toString(err).c_str() << std::endl;
		this->_zed.close();
		return 1; // Quit if an error occurred
	}

	return 0;
}

// Show camera information
// Information:
//		camera model
//		serial number
//		firmware version
//		resolution
//		fps
void CameraManager::printCameraInfo(sl::Camera &zedCameraObject){
	std::cout << "ZED Model             : " <<
			sl::toString(zedCameraObject.getCameraInformation().camera_model).c_str() << std::endl;
	std::cout << "ZED Serial Number     : " <<
			zedCameraObject.getCameraInformation().serial_number << std::endl;
	std::cout << "ZED Serial Firmware   : " <<
			zedCameraObject.getCameraInformation().firmware_version << std::endl;
	std::cout << "ZED Camera Resolution : (" <<
			(int)zedCameraObject.getResolution().width << ", " << (int)zedCameraObject.getResolution().height << ")" << std::endl;
	std::cout << "ZED Serial FPS        : " <<
			(int)zedCameraObject.getCameraFPS() << std::endl;
}

// Convert sl::Mat to cv::Mat
cv::Mat CameraManager::slMatToCvMatConverter(sl::Mat &slMat) {
		int cv_type = -1;
		switch (slMat.getDataType()){
			case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
			case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
			case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
			case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
			case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
			case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
			case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
			case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
			default: break;
		}

		return cv::Mat(
				slMat.getHeight(),
				slMat.getWidth(),
				cv_type,
				slMat.getPtr<sl::uchar1>(sl::MEM_CPU)
				);
	}

void CameraManager::setWidthAndHeight(float _widthRatio, float _heightRatio){
	sl::Resolution resolution = this->_zed.getResolution();
	this->_width = resolution.width * _widthRatio;
	this->_height= resolution.height * _heightRatio;
}

int CameraManager::cameraManagerHasLoaded(int argc, ...) {
	/*
	va_list argv;
	va_start(argv, argc);

	if(argc > 1) {
		auto pixelType = va_arg(argv, int);
	}*/
	this->parameterInitializer();
	this->openCamera();
	this->setWidthAndHeight(0.5, 0.5);

	this->printCameraInfo(this->_zed);

	sl::Mat slMat(this->_width , this->_height, sl::MAT_TYPE_8U_C4);
	cv::Mat cvMat = this->slMatToCvMatConverter(slMat);

    // Load Face cascade (.xml file)
    cv::CascadeClassifier face_cascade;
    face_cascade.load( "../data/haarcascade_frontalface_alt.xml" );

    // Detect faces
    std::vector<cv::Rect> faces;

	// Loop until 'q' is pressed
	char key = ' ';
	while (key != 'q') {

		if (this->_zed.grab() == sl::SUCCESS) {
			this->_zed.retrieveImage(slMat, sl::VIEW_LEFT, sl::MEM_CPU, this->_width, this->_height);

		    face_cascade.detectMultiScale( cvMat, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

		    // Draw circles on the detected faces
		    for( int i = 0; i < faces.size(); i++ )
		    {
		        cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
		        cv::ellipse( cvMat, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );
		    }

			cv::imshow("Image-Left", cvMat);

			// Handle key event
			key = cv::waitKey(10);
			//processKeyEvent(zed, key); // @suppress("Invalid arguments")
		}
	}

	this->_zed.close();
	return 0;
}
