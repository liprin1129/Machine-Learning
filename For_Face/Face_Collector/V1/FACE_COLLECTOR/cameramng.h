#ifndef CAMERAMNG_H
#define CAMERAMNG_H

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
//#include <chrono>

class CameraMng
{
private:
    sl::Camera zed;
    //bool isOpened;

    sl::InitParameters initParams;
    sl::RuntimeParameters runParams;
    sl::Resolution imgSize;

    // ZED Mats
    sl::Mat zedGpuMatL, zedGpuMatR;
    //sl::Mat zedMatR;

    // CV Mats
    cv::Mat cvMatL, cvMatR;

    // CV GPU Mats
    cv::cuda::GpuMat cvTmpGpuMatL, cvTmpGpuMatR;

    //std::tuple<cv::Mat, cv::Mat> cvtRLCvMatForQlabel();

public:
    CameraMng();
    ~CameraMng();

    //cv::Mat cvCpuMatL, cvCpuMatR;
    cv::cuda::GpuMat cvGpuMatL, cvGpuMatR;

    // Open the camera
    std::tuple<bool, std::string> openCamera();

    // Get one left and right CV Mats
    bool getOneFrameFromZED();

    // Convert sl::Mat to cv::Mat
    cv::Mat slGpuMat2cvMat(sl::Mat &slMat);

    // Convert sl::Mat to cv::cuda::GpuMat
    cv::cuda::GpuMat slGpuMat2cvGpuMat(sl::Mat &slMat);

    // Download GpuMat to CpuMat
    std::tuple<cv::Mat, cv::Mat> getCpuMat();
    std::tuple<cv::Mat, cv::Mat> cvtGpuMat2CpuMat(cv::cuda::GpuMat cvGpuMatL, cv::cuda::GpuMat cvGpuMatR);
    std::tuple<cv::cuda::GpuMat, cv::cuda::GpuMat> getGpuMat();
};

#endif // CAMERAMNG_H
