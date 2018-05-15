//
//  imageManager.cpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/15.
//  Copyright © 2018 170. All rights reserved.
//

#include "imageManager.hpp"

cv::Mat ImageManager::readImage(const std::string imgName, const bool colour){
	cv::Mat result;
	if (colour == false){
		result = cv::imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
	} else{
		result = cv::imread(imgName, CV_LOAD_IMAGE_COLOR);
	}
	
	//ImageHelper::showImage(result, false);
	//ImageHelper::showImage(result, true);
	//std::cout << imgName << result.size() << std::endl;
	return result;
}

void ImageManager::showImage(cv::Mat img, const bool enlarge){
	cv::Mat resizedImg;
	
	if (enlarge){
		img.copyTo(resizedImg);
		cv::resize(img, resizedImg, cv::Size(), 3, 3, cv::INTER_LINEAR );
	} else{
		img.copyTo(resizedImg);
	}
	
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	cv::imshow( "Display window", resizedImg );
	cv::waitKey(0);
}

//template <typename T>
void ImageManager::saveImageFile(std::string desFile, cv::Mat imgFile){
	cv::imwrite(desFile, imgFile);
}

cv::Mat ImageManager::resizeImage(cv::Mat img, int columns, int rows){
	cv::Mat resizedImg;// = img.clone();
	
	cv::resize(img, resizedImg, cv::Size(columns, rows), cv::INTER_AREA);
	
	return resizedImg;
}
