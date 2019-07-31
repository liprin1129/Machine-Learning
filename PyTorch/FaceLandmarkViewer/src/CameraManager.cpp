#include "CameraManager.h"

CameraManager::CameraManager() {
    
    CameraManagerHasLoaded();
}

CameraManager::~CameraManager() {
    _slLeftMat.free(sl::MEM_GPU);
    _slRightMat.free(sl::MEM_GPU);
    _cvLeftGpuMat.release();
    _cvRightGpuMat.release();

    //std::fprintf(stdout, "CameraManager deconstructor: %d pass\n", 1);
    while(_zed.isOpened()) {
        _zed.close();
    }
}

void CameraManager::openCamera()
{
    int assertCode = 0;
    // Open the camera
    sl::ERROR_CODE err = this->_zed.open(_init_params);
    if (err != sl::SUCCESS) {
        std::cout << toString(err) << std::endl;
        this->_zed.close();
        assert(assertCode!=0); // Quit if an error occurred
    }
}

// Set configuration parameters
void CameraManager::initParams()
{
    _init_params.camera_resolution = sl::RESOLUTION_HD720;
    _init_params.depth_mode = sl::DEPTH_MODE_NONE;
    _init_params.coordinate_units = sl::UNIT_MILLIMETER;
    _init_params.camera_fps = 1;
}

void CameraManager::runtimeParams() 
{
    // Set runtime parameters after opening the camera
    _runtime_parameters.sensing_mode = sl::SENSING_MODE_LAST;
    _runtime_parameters.enable_depth = false;
    _runtime_parameters.enable_point_cloud = false;
    _runtime_parameters.measure3D_reference_frame = sl::REFERENCE_FRAME_LAST;
}

void CameraManager::getOneFrameFromZED() 
{
    if (_zed.grab(_runtime_parameters) == sl::SUCCESS) {

        // Retrieve the left image, depth image in half-resolution
        _zed.retrieveImage(_slLeftMat, sl::VIEW_LEFT, sl::MEM_GPU, _zed.getResolution().width/3, _zed.getResolution().height/3);
        _zed.retrieveImage(_slRightMat, sl::VIEW_RIGHT, sl::MEM_GPU, _zed.getResolution().width/3, _zed.getResolution().height/3);

        _cvLeftGpuMat = slMatToCvMatConverterForGPU(_slLeftMat);
        _cvRightGpuMat = slMatToCvMatConverterForGPU(_slRightMat);

        _cvLeftGpuMat.download(_cvLeftMat);
        _cvRightGpuMat.download(_cvRightMat);
    }
}


void CameraManager::getFramesFromZED() {
    while (true) {
        getOneFrameFromZED();
        //std::this_thread::sleep_for(std::chrono::seconds(1));

        if(cv::waitKey(30) >= 0) {
            break;
        }
    }
}


// Convert sl::Mat to cv::cuda::GpuMat for GPU
cv::cuda::GpuMat CameraManager::slMatToCvMatConverterForGPU(sl::Mat &slMat) {
    //std::cout << "GPU SL Mat" << std::endl;

    if (slMat.getMemoryType() == sl::MEM_GPU) {
        //std::cout << "GPU SL Mat" << std::endl;
        slMat.updateCPUfromGPU();
    }

    int cvType = -1;
    switch (slMat.getDataType()){
        case sl::MAT_TYPE_32F_C1: cvType = CV_32FC1; break;
        case sl::MAT_TYPE_32F_C2: cvType = CV_32FC2; break;
        case sl::MAT_TYPE_32F_C3: cvType = CV_32FC3; break;
        case sl::MAT_TYPE_32F_C4: cvType = CV_32FC4; break;
        case sl::MAT_TYPE_8U_C1: cvType = CV_8UC1; break;
        case sl::MAT_TYPE_8U_C2: cvType = CV_8UC2; break;
        case sl::MAT_TYPE_8U_C3: cvType = CV_8UC3; break;
        case sl::MAT_TYPE_8U_C4: cvType = CV_8UC4; break;
        default: break;
    }

    return cv::cuda::GpuMat(
                static_cast<int>(slMat.getHeight()), static_cast<int>(slMat.getWidth()),
                cvType,
                slMat.getPtr<sl::uchar1>(sl::MEM_GPU), slMat.getStepBytes(sl::MEM_GPU));
}


void CameraManager::displayFrames() {
    //cv::namedWindow("left", cv::WINDOW_NORMAL);
    //cv::namedWindow("right", cv::WINDOW_NORMAL);

    while(true) {
        if (_cvLeftMat.rows > 0 && _cvLeftMat.cols > 0 && _cvRightMat.rows > 0 && _cvRightMat.cols > 0)
        {   
            //std::fprintf(stdout, "left:[%d, %d], right[%d, %d]\n", _cvLeftMat.rows, _cvLeftMat.cols, _cvRightMat.rows, _cvRightMat.cols);

            cv::imshow("left", _cvLeftMat);
            cv::imshow("right", _cvRightMat);
            //cv::imshow("left", leftCVMatWithLandmarks);
            //cv::imshow("right", rightCVMatWithLandmarks);

            if(cv::waitKey(30) >= 0) {
                break;
            }
        }

        else {
            //std::fprintf(stdout, "size error\n");
        }
    }
}


/*
void CameraManager::displayFrames(const cv::Mat &cvMat, std::string cameraPosition) {

    cv::imshow(cameraPosition, cvMat);

    cv::waitKey(0);
}
*/

void CameraManager::CameraManagerHasLoaded() {
    initParams();
    openCamera();
    //displayFrames("left");
}
