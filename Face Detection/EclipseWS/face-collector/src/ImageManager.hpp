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

protected:
	std::vector<cv::Rect> _faces;
	cv::Mat _frameCPU;
	cv::cuda::GpuMat _frameGPU, _grayGPU, _facesBufGPU;

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
	void drawRectOnFaces(cv::Mat frameCPU, std::vector<cv::Rect> faces);
	int imageManagerHasLoaded(int argc, ...);
};
#endif /* IMAGEMANAGER_HPP_ */
