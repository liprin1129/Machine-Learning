//
//  dlibFaceDetector.hpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/11.
//  Copyright © 2018 170. All rights reserved.
//

#ifndef dlibFaceDetector_hpp
#define dlibFaceDetector_hpp

#include "commonHeader.h"

#include "fileSystemManager.hpp"

class DlibFaceDetector:public FileSystemManager{
private:
	cv::Mat _refAbsImg; // this should not be changed during runtime, because it is wrapped by dlib::cv_img
	
public:
	cv::Mat readImage(const std::string imgName, const bool colour);
	void showImage(cv::Mat img, const bool enlarge);
	
	dlib::cv_image<dlib::bgr_pixel> convertMatToDlib(cv::Mat inImg);
	
	std::vector<cv::Rect> convertDlibRectToCVRect(std::vector<dlib::rectangle> dlibFaceRectVector);
	
	template <typename T>
	std::vector<dlib::rectangle> detectFace(T& dlibImg);
	
	int dlibFaceDetectorHasLoaded(int argc, ...);
};
#endif /* dlibFaceDetector_hpp */
