/*
 * ImageManager.cpp
 *
 *  Created on: May 31, 2018
 *      Author: user170
 */

#include "ImageManager.hpp"

ImageManager::ImageManager() {
	this->imageManagerHasLoaded(1, "../data/haarcascade_frontalface_alt_gpu.xml");
	//this->imageManagerHasLoaded(2, "../data/haarcascade_frontalface_alt_gpu.xml", "../data/star_wars.jpg");
	std::cout << "ImageManager constructor" << std::endl;
}

ImageManager::~ImageManager() {
	std::cout << "ImageManger Deconstructor" <<std::endl;
}

void ImageManager::imageRead(std::string absCascadeFileName, std::string absImageName) {

	cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu = cv::cuda::CascadeClassifier::create(absCascadeFileName);
	cv::Mat image_cpu = cv::imread(absImageName);

	cv::Mat image_gray;
	cv::cvtColor(image_cpu, image_gray, cv::COLOR_BGR2GRAY);

	cv::cuda::GpuMat image_gpu(image_gray);
	cv::cuda::GpuMat objbuf;
	cascade_gpu->setMinNeighbors(5);
	cascade_gpu->setScaleFactor(1.1);

	cascade_gpu->detectMultiScale(image_gpu, objbuf);
	std::vector<cv::Rect> faces;
	cascade_gpu->convert(objbuf, faces);
	std::cout << "Faces: " << faces.size() << std::endl;

	int detections_num = 4;
	for(int i = 0; i < detections_num; ++i)
	   cv::rectangle(image_cpu, faces[i], cv::Scalar(255));

	//cv::Mat image_cpu = cv::imread(absImageName);
	cv::imshow("Faces", image_cpu);
	cv::waitKey(0);
}

void ImageManager::loadCascadeClassifier(std::string absFileName) {
	this->_cascadeGPU = cv::cuda::CascadeClassifier::create(this->_cascadeFileName);
	std::cout << absFileName << " loaded" << std::endl;
}

void ImageManager::gpuParamSetup(
		double scaleFactor,
		bool findLargestObject,
		int minNeighbors,
		bool filterRects,
		bool helpScreen) {

	/*
	this->_scaleFactor = scaleFactor;
	this->_findLargestObject = findLargestObject;
	this->_filterRects = filterRects;
	this->_helpScreen = helpScreen;
	*/

	this->_cascadeGPU->setScaleFactor(scaleFactor);
	this->_cascadeGPU->setFindLargestObject(findLargestObject);
	this->_cascadeGPU->setMinNeighbors(minNeighbors);
}

void ImageManager::uploadFrameToGPU(cv::Mat frameCPU) {
	// Convert RGBA
	//cv::Mat grayFrame;
	//cv::cvtColor(capturedFrame, grayFrame, cv::COLOR_RGBA2GRAY);

	frameCPU.copyTo(this->_frameCPU);
	// Upload captured frame on CPU memory to GPU memory
	this->_frameGPU.upload(this->_frameCPU);
}

void ImageManager::convertRGBAToGrayGPU(cv::cuda::GpuMat rgbaFrame){
	// Change RGBA frame to gray frame
	cv::cuda::cvtColor(rgbaFrame, this->_grayGPU, cv::COLOR_RGBA2GRAY);
}

void ImageManager::startCascadeFaceDetection(cv::cuda::GpuMat){
	// Upload CPU frame to GPU memory

	this->_cascadeGPU->detectMultiScale(this->_grayGPU, this->_facesBufGPU);
    this->_cascadeGPU->convert(this->_facesBufGPU, this->_faces);

    //std::cout << std::setfill(' ') << std::setprecision(2);
    //std::cout << std::setw(6) << this->_faces.size() << std::endl;
}

void ImageManager::drawRectOnFaces(cv::Mat frameCPU, std::vector<cv::Rect> faces, int thick = 1){
	if (faces.size() != 0){
		for(int i = 0; i < faces.size(); ++i)
		   cv::rectangle(frameCPU, faces[i], cv::Scalar(66, 66, 244), thick);
	}
}

void ImageManager::getFaces(cv::Mat capturedFrame){
	this->uploadFrameToGPU(capturedFrame);
	this->convertRGBAToGrayGPU(this->_frameGPU);
	this->startCascadeFaceDetection(this->_grayGPU);
	this->drawRectOnFaces(this->_frameCPU, this->_faces, 3);

	//return this->_faces;
}

cv::Mat ImageManager::truncateFirstFace(cv::Mat frame, std::vector<cv::Rect> faceCvRectVector) {
	/*
	for (std::vector<std::string>::const_iterator iterFace=faceCvRectVector.begin(); iterFace!=faceCvRectVector.end(); ++iterFace){
		auto iterFaceIndex = iterFace - faceCvRectVector.begin();

		cv::Rect faceCVRect = faceCvRectVector[iterFaceIndex];
	}
	*/

	cv::Rect faceCvRect = faceCvRectVector.front();
	return frame(faceCvRect);
}

int ImageManager::imageManagerHasLoaded(int argc, ...) {
	// Read multiple arguments
	va_list argv;
	va_start(argv, argc);

	/*//Test for GPU haar cascade face detection
	std::string cascadeName = va_arg(argv, char*);
	std::string imgName = va_arg(argv, char*);
	std::cout << cascadeName << ", " << imgName << std::endl;
	this->imageRead(cascadeName, imgName);
	*/

	// Print out gpu device information
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	// Assign cascade file name
	this->_cascadeFileName = va_arg(argv, char*);

	// Instance of cv::cuda::CascadeClassifier
	this->loadCascadeClassifier(this->_cascadeFileName);

    // GPU parameters
	this->gpuParamSetup(1.2, true, 4, true, false);

	return 0;
}
