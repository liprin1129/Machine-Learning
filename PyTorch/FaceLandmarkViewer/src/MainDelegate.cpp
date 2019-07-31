#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){

    //std::unique_ptr<CameraManager> cm(new CameraManager);
    CameraManager cm;
    FaceLandmarksDetector fld;

    // Load Face Detector
    CascadeClassifier faceDetector(argv[1]);
    
    // Create an instance of Facemark
    cv::face::FacemarkLBF::Params params;
    params.model_filename = argv[2];//_faceModelPath;
    cv::Ptr<cv::face::FacemarkLBF> facemark = cv::face::FacemarkLBF::create(params);
    facemark->loadModel(params.model_filename);
    
    // Variable to store a video frame and its grayscale 
    cv::Mat frame, gray;

    cv::namedWindow("Facial Landmark Detection", cv::WINDOW_NORMAL);

    
    while(true)
    {   
        cm.getOneFrameFromZED();
        frame = cm.getCVLeftMat();

        if (frame.cols > 0 and frame.rows) {

            fld.landmarkDetector(facemark, faceDetector, frame);

            // Display results
            cv::imshow("Facial Landmark Detection", frame);

            // Exit loop if ESC is pressed
            if (waitKey(1) == 27) {
                cm.~CameraManager();
                
                delete facemark;
                break;
            }
        }
    }
    
    return 0;
}
