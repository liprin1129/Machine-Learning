#include "FaceLandmarksDetector.h"

std::vector<std::tuple<float, float>> FaceLandmarksDetector::landmarkDetector(const cv::Ptr<cv::face::Facemark> &facemark, cv::CascadeClassifier &faceDetector, cv::Mat &inCVMat) {
    std::vector<std::tuple<float, float>> landmarksVec2f;

    if (inCVMat.cols > 0 and inCVMat.rows) {
        // Find face
        
        // Convert frame to grayscale because
        // faceDetector requires grayscale image.
        cv::cvtColor(inCVMat, _gray, cv::COLOR_BGR2GRAY);

        // Detect faces
        faceDetector.detectMultiScale(_gray, _faces);
        
        // Variable for landmarks. 
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we 
        // use a vector of vector of points. 
        //std::vector< std::vector<cv::Point2f> > landmarks;
        
        // Run landmark detector
        bool success = facemark->fit(inCVMat, _faces, _landmarksCVPint2f);
        
        if(success)
        {
            // If successful, render the landmarks on the face
            for(int i = 0; i < _landmarksCVPint2f.size(); i++)
            {
                DrawLandmarks::drawLandmarks(inCVMat, _landmarksCVPint2f[i]);
            }
            
            for(auto const &landmark: _landmarksCVPint2f[0]) {
                //_landmarksVec2f.push_back(std::make_tuple(landmark.x, landmark.y));
                landmarksVec2f.push_back(std::make_tuple(landmark.x, landmark.y));
                //std::fprintf(stdout, "[%f, %f]\n", landmark.x, landmark.y);
            }
            //std::cout << std::endl;
        }
    }

    return landmarksVec2f;
}

void FaceLandmarksDetector::FaceLandmarksDetectorHasLoaded() {
    // Load Face Detector
    //_faceDetector = cv::CascadeClassifier(_haarCascadePath);
    
    // Create an instance of Facemark
    cv::face::FacemarkLBF::Params params;
    std::cout << "pass 1" << std::endl;
    params.model_filename = _faceModelPath;
    std::cout << "pass 2" << std::endl;
    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create(params);
    std::cout << "pass 3" << std::endl; 
}