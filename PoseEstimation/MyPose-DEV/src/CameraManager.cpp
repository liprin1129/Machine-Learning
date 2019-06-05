#include "CameraManager.h"

CameraManager::CameraManager() {
// Constructor function for CameraManager class:
    initParams();
    runtimeParams();
    
    std::cout << "CameraManager Class Instace" << std::endl;
}


CameraManager::~CameraManager() {
// Destructor function for CameraManager class:
    //_zed.close();

    std::cout << "CameraManager Class Destroyed" << std::endl;
}


void CameraManager::initParams(){
// Initial camera configs:
    _init_params.camera_resolution = sl::RESOLUTION_HD720;
    _init_params.depth_mode = sl::DEPTH_MODE_NONE;
    _init_params.coordinate_system = sl::COORDINATE_SYSTEM_IMAGE;
    //_init_params.coordinate_units = sl::UNIT_METER;

    if(_init_params.camera_resolution == 0) {
        focalLength=1400;
    } 
    else if (_init_params.camera_resolution == 1) {
        focalLength=1400;
    }
    else if (_init_params.camera_resolution == 2) {
        focalLength=700;
    }
    else if (_init_params.camera_resolution == 3) {
        focalLength=350;
    }
}


void CameraManager::runtimeParams() {
// Runtime camera configs:
    _runtime_params.sensing_mode = sl::SENSING_MODE_LAST;
}


int CameraManager::cameraOpen() {
    sl::ERROR_CODE errCode = _zed.open(_init_params);
    
    imgViewWidth = _zed.getResolution().width; // get width
    imgViewHeight = _zed.getResolution().height; // and height
    
    if (errCode != sl::SUCCESS) {
        std::cout << sl::toString(errCode).c_str() << std::endl;
        _zed.close();
        return 1;
    }
}


cv::Mat CameraManager::getOneFrame() {
// Conversion function between sl::Mat and cv::Mat:
    _zed.retrieveImage(slMat, sl::VIEW_SIDE_BY_SIDE, sl::MEM_CPU);
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(slMat.getHeight(), slMat.getWidth(), CV_8UC4, slMat.getPtr<sl::uchar1>(sl::MEM_CPU));
}


std::vector<double> CameraManager::getJointZ() {
    //std::vector<std::tuple<double, double>> jointXY;

    // Pseudo points
    auto rightWrist = std::make_tuple( // tuple for joint pixel locations in left and right images repectively
        std::make_tuple(456, 448), 
        std::make_tuple(598, 448));

    auto [rightWristLImg, rightWristRImg] = rightWrist;

    auto [rwLX, rwLY] = rightWristLImg;
    auto [rwRX, rwRY] = rightWristRImg;
    std::cout << '(' << rwLX << ", " << rwLY << ')' << std::endl;
    std::cout << '(' << rwRX << ", " << rwRY << ')' << std::endl;

    auto leftWrist = std::make_tuple( // tuple for joint pixel locations in left and right images repectively
        std::make_tuple(729, 511), 
        std::make_tuple(846, 511));

    auto [leftWristLImg, leftWristRImg] = leftWrist;

    auto [lwLX, lwLY] = leftWristLImg;
    auto [lwRX, lwRY] = leftWristRImg;
    std::cout << '(' << lwLX << ", " << lwLY << ')' << std::endl;
    std::cout << '(' << lwRX << ", " << lwRY << ')' << std::endl;

    /*
    // Get calibration information
    auto calibInfoL = _zed.getCameraInformation().calibration_parameters.left_cam;
    std::cout << "<Left>" << std::endl;
    std::cout << calibInfoL.fx << std::endl;
    std::cout << calibInfoL.fy << std::endl;
    std::cout << calibInfoL.cx << std::endl;
    std::cout << calibInfoL.cy << std::endl;
    std::cout << calibInfoL.v_fov << std::endl;
    std::cout << calibInfoL.h_fov << std::endl;
    std::cout << calibInfoL.d_fov << std::endl;
    std::cout << imgViewWidth << std::endl;
    std::cout << imgViewHeight << std::endl;

    auto calibInfoR = _zed.getCameraInformation().calibration_parameters.right_cam;
    std::cout << "<Right>" << std::endl;
    std::cout << calibInfoR.fx << std::endl;
    std::cout << calibInfoR.fy << std::endl;
    std::cout << calibInfoR.cx << std::endl;
    std::cout << calibInfoR.cy << std::endl;
    std::cout << calibInfoR.v_fov << std::endl;
    std::cout << calibInfoR.h_fov << std::endl;
    std::cout << calibInfoR.d_fov << std::endl;
    std::cout << imgViewWidth << std::endl;
    std::cout << imgViewHeight << std::endl;
    */

   std::cout << _init_params.camera_resolution << ": " << focalLength << std::endl;
   auto rwLZ = static_cast<double>(120)*focalLength/(rwRX-rwLX);
   printf("right jointZ: %f\n", rwLZ);

   std::cout << _init_params.camera_resolution << ": " << focalLength << std::endl;
   auto lwLZ = static_cast<double>(120)*focalLength/(lwRX-lwLX);
   printf("left jointZ: %f\n", lwLZ);
}
int CameraManager::cameraManagerDidLoad() {
// main method for CameraManger class:
    
    auto errInt = 0; // normal condition is 0, abnormal is any other than 0.
    /*
    if (cameraOpen() != errInt) {
        std::cout << "Camera can't open" << std::endl;
        return 1;
    }
    */
    /*
    while (1) { // take only one frame
        if (_zed.grab(_runtime_params) == sl::SUCCESS) {
            cvMat = getOneFrame();
            cv::Mat cvLeftMat(cvMat, cv::Rect(0, 0, imgViewWidth, imgViewHeight));
            cv::Mat cvRightMat(cvMat, cv::Rect(imgViewWidth, 0, imgViewWidth, imgViewHeight));

            cv::imwrite("test.jpg", cvMat);
            cv::imwrite("left.jpg", cvLeftMat);
            cv::imwrite("right.jpg", cvRightMat);
            break;
        }
    }
    */

    getJointZ();

    // JSON test
    JsonFileManager jfm;
    jfm.jsonPrint();

    return errInt;
}