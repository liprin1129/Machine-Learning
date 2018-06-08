/*
 * ImageManager.hpp
 *
 *  Created on: May 31, 2018
 *      Author: user170
 */

#ifndef IMAGEMANAGER_HPP_
#define IMAGEMANAGER_HPP_

#include "CommonHeaders.hpp"

class ImageManager{

private:
	std::string _cascadeFileName;

	// Instance of cv::cuda::CascadeClassifier
	cv::Ptr<cv::cuda::CascadeClassifier> _cascadeGPU;

	cv::cuda::GpuMat _frameGPU, _grayGPU, _facesBufGPU;

protected:
	cv::Mat _frameCPU;
	std::vector<cv::Rect> _faces;
	/*
	// OpenCV GPU Parameters
	bool _useGPU;
	double _scaleFactor;
	bool _findLargestObject;
	bool _filterRects;
	bool _helpScreen;
	*/

public:
	ImageManager();
	virtual ~ImageManager();

	//std::vector<cv::Rect> faces() const {return _faces;}

	void imageRead(std::string absCascadeFileName, std::string absImageName);

	// Instantiate cv::cuda::CascadeClassifier
	// Arguments:
	//		absFileName: (std::string) haar cascade file name
	void loadCascadeClassifier(std::string absFileName);

	void gpuParamSetup(
			double scaleFactor,
			bool findLargestObject,
			int minNeighbors,
			bool filterRects,
			bool helpScreen);

	void uploadFrameToGPU(cv::Mat capturedFrame);
	void convertRGBAToGrayGPU(cv::cuda::GpuMat rgbaFrame);
	void startCascadeFaceDetection(cv::cuda::GpuMat);
	void drawRectOnFaces(cv::Mat frameCPU, std::vector<cv::Rect> faces, int thick);
	//std::vector<cv::Rect> getFaces(cv::Mat frameCPU);
	void getFaces(cv::Mat frameCPU);

	cv::Mat truncateFirstFace(cv::Mat frame, std::vector<cv::Rect> faceVector);
	int imageManagerHasLoaded(int argc, ...);
};
#endif /* IMAGEMANAGER_HPP_ */
