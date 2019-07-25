#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
    std::shared_ptr<CameraManager> cmPtr(new CameraManager);
    std::shared_ptr<TrainerInferrer> tiPtr(new TrainerInferrer);
    
    FaceLandmarkNet fln(3, false);
    fln->eval();    // evaluation mode
    fln->to(torch::Device(torch::kCUDA));   // Upload the model to the device {CPU, GPU}
    
    cmPtr->getOneFrameFromZED();
    tiPtr->inferStereo(fln, cmPtr->getCVLeftMat(), cmPtr->getCVRightMat(), torch::Device(torch::kCUDA));

    //cv::namedWindow("left", cv::WINDOW_NORMAL);
    //cv::namedWindow("right", cv::WINDOW_NORMAL);
    
    //std::thread getFrames(&CameraManager::getFramesFromZED, cmPtr);
    //std::thread display(&CameraManager::displayFrames, cmPtr, cmPtr->getCVLeftMat(), cmPtr->getCVRightMat());
    //std::thread infer(&TrainerInferrer::inferStereo, tiPtr, fln, cmPtr->getCVLeftMat(), cmPtr->getCVRightMat(), torch::Device(torch::kCUDA));
    //tiPtr->inferStereo(fln, cmPtr->getCVLeftMat(), cmPtr->getCVRightMat(), torch::Device(torch::kCUDA));
    
    //getFrames.join();
    //display.join();
    //infer.join();
}