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
#include "imageManager.hpp"

class DlibFaceDetector:public FileSystemManager, public ImageManager{
	
public:
	dlib::cv_image<dlib::bgr_pixel> convertMatToDlib(cv::Mat inImg);
	std::vector<cv::Rect> convertDlibRectToCVRect(std::vector<dlib::rectangle> dlibFaceRectVector);
	
	template <typename T>
	std::vector<dlib::rectangle> detectFace(T& dlibImg);
	
	int dlibFaceDetectorHasLoaded(int argc, ...);
};
#endif /* dlibFaceDetector_hpp */
