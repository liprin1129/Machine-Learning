#include "CameraManager.h"

CameraManager::CameraManager() {
// Constructor function for CameraManager class
    initParams();
    std::cout << "Constructor" << std::endl;
}


void CameraManager::initParams(){
// Initial camera configs
    _init_params.camera_resolution = sl::RESOLUTION_HD1080;
    _init_params.depth_mode = sl::DEPTH_MODE_NONE;
    _init_params.coordinate_system = sl::COORDINATE_SYSTEM_IMAGE;
    //_init_params.coordinate_units = sl::UNIT_METER;
}


void CameraManager::runtimeParams() {
// Runtime camera configs
    _runtime_params.sensing_mode = sl::SENSING_MODE_LAST;
}


int CameraManager::cameraOpen() {
    sl::ERROR_CODE errCode = _zed.open(_init_params);
    if (errCode != sl::SUCCESS) {
        std::cout << sl::toString(errCode).c_str() << std::endl;
        _zed.close();
        return 1;
    }
}


void CameraManager::slMat2cvMatBridge(){
// Pointer for slMat to cvMat
    slMat = sl::Mat(_zed.getResolution().width, _zed.getResolution().height, sl::MAT_TYPE_8U_C4);
    cvMat = slMat2cvMatConverter(slMat);
}


cv::Mat CameraManager::slMat2cvMatConverter(sl::Mat& input) {
// Conversion function between sl::Mat and cv::Mat
    auto cv_type = CV_8UC4;
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}


// main method for CameraManger class
int CameraManager::cameraManagerDidLoad() {
    auto errInt = 0;

    if (cameraOpen() == 1) {
        std::cout << "Camera can't open" << std::endl;
        return 1;
    }
    runtimeParams();

    _zed.close();

    return 0;
}