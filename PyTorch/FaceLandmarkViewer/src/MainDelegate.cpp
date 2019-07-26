#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
    // Set up webcam for video capture
    //VideoCapture cam(0);
    std::shared_ptr<CameraManager> cm(new CameraManager);

    FaceLandmarksDetector fld;

    // Load Face Detector
    CascadeClassifier faceDetector(argv[1]);
    
    // Create an instance of Facemark
    cv::face::FacemarkLBF::Params params;
    params.model_filename = argv[2];//_faceModelPath;
    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create(params);
    facemark->loadModel(params.model_filename);

    // Variable to store a video frame and its grayscale 
    cv::Mat frame, gray;

    while(true)
    {   
        cm->getOneFrameFromZED();
        frame = cm->getCVLeftMat();

        if (frame.cols > 0 and frame.rows) {

            fld.landmarkDetector(facemark, faceDetector, frame);

            // Display results 
            imshow("Facial Landmark Detection", frame);
            // Exit loop if ESC is pressed
            if (waitKey(1) == 27) break;
        }
    }

    /*
    // Load Face Detector
    CascadeClassifier faceDetector(argv[1]);
    
    // Create an instance of Facemark
    cv::face::FacemarkLBF::Params params;
    params.model_filename = argv[2];//_faceModelPath;
    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create(params);
    facemark->loadModel(params.model_filename);

    // Variable to store a video frame and its grayscale 
    cv::Mat frame, gray;

    while(true)
    {   
        cm->getOneFrameFromZED();
        frame = cm->getCVLeftMat();

        if (frame.cols > 0 and frame.rows) {

        // Find face
        vector<Rect> faces;
        // Convert frame to grayscale because
        // faceDetector requires grayscale image.
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        faceDetector.detectMultiScale(gray, faces);

        // Variable for landmarks. 
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we 
        // use a vector of vector of points. 
        vector< vector<Point2f> > landmarks;

        // Run landmark detector
        bool success = facemark->fit(frame,faces,landmarks);

        if(success)
        {
        // If successful, render the landmarks on the face
            for(int i = 0; i < landmarks.size(); i++)
            {
                DrawLandmarks::drawLandmarks(frame, landmarks[i]);
            }
        }

        // Display results 
        imshow("Facial Landmark Detection", frame);
        // Exit loop if ESC is pressed
        if (waitKey(1) == 27) break;
        }
    }
    */
    
    return 0;
}
