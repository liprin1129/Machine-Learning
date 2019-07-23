#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
    std::shared_ptr<CameraManager> cmPtr(new CameraManager);

    cv::namedWindow("left", cv::WINDOW_NORMAL);
    cv::namedWindow("right", cv::WINDOW_NORMAL);
    
    std::thread getFrames(&CameraManager::getFramesFromZED, cmPtr);
    std::thread display(&CameraManager::displayFrames, cmPtr);

    getFrames.join();
    display.join();
}