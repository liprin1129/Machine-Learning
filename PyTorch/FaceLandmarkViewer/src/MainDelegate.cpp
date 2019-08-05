#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){

    //std::unique_ptr<CameraManager> cm(new CameraManager);
    CameraManager cm;
    FaceLandmarksDetector fld;
    DepthEstimator de;

    // Load Face Detector
    cv::CascadeClassifier faceDetector(argv[1]);
    
    // Create an instance of Facemark
    cv::face::FacemarkLBF::Params params;
    params.model_filename = argv[2];//_faceModelPath;
    cv::Ptr<cv::face::FacemarkLBF> facemark = cv::face::FacemarkLBF::create(params);
    facemark->loadModel(params.model_filename);
    
    // Variable to store a video frame and its grayscale 
    cv::Mat lFrame, lGray, rFrame, rGray;

    cv::namedWindow("Right Facial Landmark Detection", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Left Facial Landmark Detection", cv::WINDOW_AUTOSIZE);
    //std::cout << "get frame" << std::endl;
    std::cout << std::endl;

    int initCount = 0;
    if (std::string(argv[3]) == "Register") {
        std::cout << "Register\n";
        initCount = 500;
    }
    else if (std::string(argv[3]) == "Predict") {
        std::cout << "Predict\n";
        initCount = 80;
    }

    int clockCount = initCount;

    while(true)
    {   
        cm.getOneFrameFromZED(); // get a left and right camera frame from the camera
        lFrame = cm.getCVLeftMat(); // return a left camera frame
        rFrame = cm.getCVRightMat(); // return a right camera frame

        if (lFrame.cols > 0 and lFrame.rows > 0 and rFrame.cols > 0 and rFrame.rows > 0) {
            // Calculate landmarks
            auto lLandmarks = fld.landmarkDetector(facemark, faceDetector, lFrame);
            auto rLandmarks = fld.landmarkDetector(facemark, faceDetector, rFrame);

            // Calculate 3D coordinates
            if (lLandmarks.size() > 0 and rLandmarks.size() > 0) {
                auto [lfx, lfy] = cm.getLeftFocalLength();
                auto [lcx, lcy] = cm.getLeftCameraOpticalCentre();
                de.estimateCoordinates3D(lLandmarks, rLandmarks, lfx, lfy, lcx, lcy, cm.getCameraFocalLength());

                //de.incrementalMeanAndCovariance();
                de.incrementalMean(clockCount);
            }

            // Display results
            cv::moveWindow("Left Facial Landmark Detection", 20,20);
            cv::moveWindow("Right Facial Landmark Detection", 20+lFrame.cols,20);
            cv::imshow("Left Facial Landmark Detection", lFrame);
            cv::imshow("Right Facial Landmark Detection", rFrame);
            int countPercent = ((initCount-clockCount)*100)/(initCount);
            std::cout << "\r" << countPercent << " %" << std::flush;
            //std::cout << "\r" << clockCount << " %" << std::flush;
            

            // Exit loop if ESC is pressed
            if (cv::waitKey(1) == 27 or countPercent >= 100) {
                cv::destroyAllWindows();
                cm.~CameraManager();
                
                if (std::string(argv[3]) == "Register") {
                    std::cout << "Enter you employee number\n";
                    std::string employNum;
                    std::cin >> employNum;
                    // Save means and variances to CSV file
                    DataWrangling::Utilities::writeMeanAndVarTensorToCSV(de.getMeanTensor(), employNum);
                }
                else if (std::string(argv[3]) == "Predict") {
                    FaceRecognizer::landmarkMeanDifferences(de.getMeanTensor());
                }
                
                delete facemark;
                break;
            }
        }
    }

    cm.~CameraManager();

    return 0;
}
