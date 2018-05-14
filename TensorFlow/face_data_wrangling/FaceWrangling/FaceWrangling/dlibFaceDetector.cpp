//
//  dlibFaceDetector.cpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/11.
//  Copyright © 2018 170. All rights reserved.
//

#include "dlibFaceDetector.hpp"

cv::Mat DlibFaceDetector::readImage(const std::string imgName, const bool colour){
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

void DlibFaceDetector::showImage(cv::Mat img, const bool enlarge){
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

dlib::cv_image<dlib::bgr_pixel> DlibFaceDetector::convertMatToDlib(cv::Mat inImg){
	dlib::cv_image<dlib::bgr_pixel> cimg(inImg);
	
	return cimg;
}

template <typename T> void DlibFaceDetector::saveImageFile(std::string desFile, T& imgFile){
	cv::imwrite(desFile, imgFile);
}

template <typename T>
std::vector<dlib::rectangle> DlibFaceDetector::detectFace(T &dlibImg){
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	
	std::vector<dlib::rectangle> face = detector(dlibImg);
	/*
	std::cout << "Number of faces detected: " << face.size() << std::endl;
	
	dlib::image_window win;
	
	win.clear_overlay();
	win.set_image(dlibImg);
	win.add_overlay(face, dlib::rgb_pixel(255,0,0));
	std::cout << "Hit enter to process the next image..." << std::endl;
	std::cin.get();
	*/
	
	return face;
}

std::vector<cv::Rect> DlibFaceDetector::convertDlibRectToCVRect(std::vector<dlib::rectangle> dlibFaceRectVector){
	std::vector<cv::Rect> cvRectVector;
	
	for (std::vector<dlib::rectangle>::const_iterator iter=dlibFaceRectVector.begin(); iter!=dlibFaceRectVector.end(); ++iter){
		auto iterIndex = iter-dlibFaceRectVector.begin();
		/* //Print dlib::rectangle
		 std::cout << iterIndex << face[iterIndex].left() << '|' << face[iterIndex].right() <<std::endl;
		 std::cout << iterIndex << face[iterIndex].top() << '|' << face[iterIndex].bottom() <<std::endl;
		 */
		// Convert dlib::rectangle to cv::rectangle
		auto rectCV = cv::Rect(cv::Point2l(dlibFaceRectVector[iterIndex].left(), dlibFaceRectVector[iterIndex].top()),
							   cv::Point2l(dlibFaceRectVector[iterIndex].right(), dlibFaceRectVector[iterIndex].bottom()));
		cvRectVector.push_back(rectCV);
		//std::cout << "CV::Rect :" << rectCV << std::endl;
	}
	
	return cvRectVector;
}

int DlibFaceDetector::dlibFaceDetectorHasLoaded(int argc, ...){
	va_list argv;
	va_start(argv, argc);
	
	if(argc > 1){
		auto dirPath = va_arg(argv, char*);
		this->fileInvestigator(dirPath, ".jpg");
		
		// Read a image
		for (std::vector<std::string>::const_iterator iterPath=_allFileAbsPath.begin(); iterPath!=_allFileAbsPath.end(); ++iterPath){
			
			auto iterPathIndex = iterPath - _allFileAbsPath.begin();
			
			if (!boost::filesystem::exists(savePath+_fileName[iterPathIndex])){ // Check if expected output file is exist
				std::cout << "_allFileAbsPath path:" << _allFileAbsPath[iterPathIndex] << std::endl;
				_refAbsImg = this->readImage(_allFileAbsPath[iterPathIndex], true);
				//std::cout << "_allFileAbsPath size: " <<(int)_allFileAbsPath.size() << std::endl;
				//std::cout << "_refAbsImg shape" << _refAbsImg.size << std::endl;
				//this->showImage(_refAbsImg, true);
				
				// Convert OpenCV Mat to Dlib (actually it just wraps)
				auto dlibImg = this->convertMatToDlib(_refAbsImg);
				// Face detection
				auto faceDlibRectVec = this->detectFace(dlibImg);
				// Convert dlib::rectangle to cv::Rect
				auto faceCVRectVec = this->convertDlibRectToCVRect(faceDlibRectVec);
				
				if (!faceCVRectVec.empty()){
					// Print out how many faces were detected
					for (std::vector<cv::Rect>::const_iterator iter=faceCVRectVec.begin(); iter!=faceCVRectVec.end(); ++iter){
						auto iterFaceIndex = iter-faceCVRectVec.begin();
						//std::cout << iterFaceIndex << " | " << *iter << std::endl;
						
						// Check face image size is inside the original image
						auto faceCVRect = faceCVRectVec[iterFaceIndex];
						if ((faceCVRect.x+faceCVRect.width <= _refAbsImg.rows) && (faceCVRect.y+faceCVRect.height <= _refAbsImg.cols)) {
							// Truncat a face from the image
							auto faceImgCV = _refAbsImg(faceCVRectVec[iterFaceIndex]);
							auto desAbsFileName = savePath+_fileName[iterPathIndex];
							desAbsFileName.insert(desAbsFileName.size()-4, "_"+std::to_string(iterFaceIndex));
							
							//std::cout << desAbsFileName << std::endl;
							this->saveImageFile(desAbsFileName, faceImgCV);
						}
					}
				}
			}
			
			if ((iterPathIndex % 100) == 0){
				//std::cout << iterPathIndex << " | " << _allFileAbsPath.size() << std::endl;
				std::cout << "\n\n" << iterPathIndex / (float)_allFileAbsPath.size() <<" %" << "\n\n";
			}
		}

		/*
		_refAbsImg = this->readImage("/Users/user170/Developments/Personal-Dev./Machine-Learning/Data/Face/lfw/Liza_Minnelli/Liza_Minnelli_0002.jpg", true);

		//this->showImage(_refAbsImg, true);
		
		// Convert OpenCV Mat to Dlib (actually it just wraps)
		auto dlibImg = this->convertMatToDlib(_refAbsImg);
		// Face detection
		auto faceDlibRectVec = this->detectFace(dlibImg);
		// Convert dlib::rectangle to cv::Rect
		auto faceCVRectVec = this->convertDlibRectToCVRect(faceDlibRectVec);
 
		// Print out how many faces were detected
        for (std::vector<cv::Rect>::const_iterator iter=faceCVRectVec.begin(); iter!=faceCVRectVec.end(); ++iter){
            auto iterIndex = iter-faceCVRectVec.begin();
            //std::cout << iterIndex << " | " << *iter << std::endl;
			
			std::cout << _refAbsImg.rows << '|' << _refAbsImg.cols << std::endl;
			std::cout << faceCVRectVec[iterIndex] << std::endl;
			
			// Check face image size is inside the original image
			auto faceCVRect = faceCVRectVec[iterIndex];
			std::cout << faceCVRect.x+faceCVRect.width << '|' << faceCVRect.y+faceCVRect.height << std::endl << std::endl;
			
			if ((faceCVRect.x+faceCVRect.width <= _refAbsImg.rows) &&
				(faceCVRect.y+faceCVRect.height <= _refAbsImg.cols) &&
				(faceCVRect.x >= 0) &&
				(faceCVRect.y >= 0)) {
				// Truncat a face from the image
				auto faceImgCV = _refAbsImg(faceCVRectVec[iterIndex]);
				this->saveImageFile(savePath+_fileName[iterIndex], faceImgCV);
			}
        }*/
	}
	
	va_end(argv);
	return 0;
}

/*
int DlibFaceDetector::dlibFaceDetectorHasLoaded(int argc, ...){
    va_list argv;
    va_start(argv, argc);
    
    if(argc > 1){
        auto dirPath = va_arg(argv, char*);
        this->fileInvestigator(dirPath, ".jpg");
        
        _refAbsImg = this->readImage(_allFileAbsPath[0], true);
        std::cout << _allFileAbsPath.size() << std::endl;
        std::cout << _allFileAbsPath[0] << std::endl;
        std::cout << "_refAbsImg shape: " << _refAbsImg.size << std::endl;
    }
    
    va_end(argv);
    return 0;
}
*/
