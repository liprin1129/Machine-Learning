#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"

class FaceLandmarksDetector {
    private:
        std::string _haarCascadePath;
        std::string _faceModelPath;

        cv::Mat _frame, _gray;

        std::vector<cv::Rect> _faces;
        std::vector<std::vector<cv::Point2f>> _landmarksCVPint2f;
        std::vector<std::tuple<float, float>> _landmarksTuple2f;
        //cv::CascadeClassifier _faceDetector;
        //cv::Ptr<cv::face::Facemark> _facemark;

    public:
        std::vector<std::tuple<float, float>> getLandmarksTuple2f() {return _landmarksTuple2f;};
        
        void landmarkDetector(const cv::Ptr<cv::face::Facemark> &facemark, cv::CascadeClassifier &faceDetector, cv::Mat &inCVMat);
        void FaceLandmarksDetectorHasLoaded();
};