#ifndef FACELANDMARKER_H
#define FACELANDMARKER_H

//#include <iostream>
#include <opencv2/opencv.hpp>

class FaceLandmarker
{
private:
    //cv::cuda::GpuMat tmpGrayL, tmpGrayR, tmpObjL, tmpObjR;
    //Detected faces cv::Rect vector
    std::vector<cv::Rect> facesL, facesR;
    cv::Mat cvCpuMatL, cvCpuMatR;

    // Opencv Haar cascade
    cv::Ptr<cv::cuda::CascadeClassifier> gpuCascade;
    void cudaCascadeParamSetup(double scaleFactor, bool findLargestObject,
                               int minNeighbors, bool filterRects);
    void drawRectOnFaces(cv::cuda::GpuMat cvGpuMatL, cv::cuda::GpuMat cvGpuMatR, int thickness);

    int count;
public:
    //int faceLocX;
    //int faceLocY;
    cv::Rect faceL;

    FaceLandmarker();
    ~FaceLandmarker();
    std::tuple<cv::Mat, cv::Mat> findFaces(cv::cuda::GpuMat cvGpuMatL, cv::cuda::GpuMat cvGpuMatR);
};

#endif // FACELANDMARKER_H
