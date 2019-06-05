#include <sl_zed/Camera.hpp>
#include <opencv2/opencv.hpp>
#include "JsonFileManager.h"

class CameraManager {
    private:
        // Instance variables
        sl::Camera _zed;
        sl::InitParameters _init_params;
        sl::RuntimeParameters _runtime_params;

        int imgViewWidth;
        int imgViewHeight;
        int focalLength;

        sl::Mat slMat;
        cv::Mat cvMat;

        // Methods
        void initParams(); // Initial camera configs
        void runtimeParams(); // Runtime camera configs
        int cameraOpen();

        cv::Mat getOneFrame();
    public:
        // Constructor function for CameraManager class
        CameraManager();
        ~CameraManager();
        std::vector<double> getJointZ();

        // main method for CameraManger class
        int cameraManagerDidLoad();
};