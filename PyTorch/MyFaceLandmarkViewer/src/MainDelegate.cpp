#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
    std::shared_ptr<CameraManager> cmPtr(new CameraManager);
    //std::shared_ptr<TrainerInferrer> tiPtr(new TrainerInferrer);
    
    cv::namedWindow("left", cv::WINDOW_NORMAL);
    cv::namedWindow("right", cv::WINDOW_NORMAL);

    //auto leftImage = cv::imread("/DATASETs/Face/Face-SJC/croped-left.png", cv::IMREAD_UNCHANGED);
    //auto rightImage = cv::imread("/DATASETs/Face/Face-SJC/croped-right.png", cv::IMREAD_UNCHANGED);
    
    std::thread getFrames(&CameraManager::getFramesFromZED, cmPtr);
    std::thread display(&CameraManager::displayFrames, cmPtr);
    //std::thread infer(&TrainerInferrer::inferStereo, tiPtr, fln, cmPtr->getCVLeftMat(), cmPtr->getCVRightMat(), torch::Device(torch::kCUDA));
    //tiPtr->inferStereo(fln, cmPtr->getCVLeftMat(), cmPtr->getCVRightMat(), torch::Device(torch::kCUDA));
    
    getFrames.join();
    display.join();
    //infer.join();
}