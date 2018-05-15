//
//  imageManager.hpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/15.
//  Copyright © 2018 170. All rights reserved.
//

#ifndef imageManager_hpp
#define imageManager_hpp

#include "commonHeader.h"

class ImageManager{
protected:
	cv::Mat _refAbsImg; // this should not be changed during runtime, because it is wrapped by dlib::cv_img
	
	int _outColoumns = 32;
	int _outRows = 32;
	
public:
	// 170 Mac
	std::string savePath = "/Users/user170/Developments/Personal-Dev./Machine-Learning/Data/Face/ifw_truncated/";
	
	// 170 Ubuntu
	// std::string savePath = "";
	
	// Pure Mac
	// std::string savePath = "/Users/pure/Developments/Personal-Study/Machine-Learning/Data/lfw_truncated/";
	
	cv::Mat readImage(const std::string imgName, const bool colour);
	void showImage(cv::Mat img, const bool enlarge);
	
	// Write an image file
	//template <typename T>
	void saveImageFile(std::string desFile, cv::Mat imgFile);
	
	// Resize an image
	cv::Mat resizeImage(cv::Mat img, int columns, int rows);
};
#endif /* imageManager_hpp */
