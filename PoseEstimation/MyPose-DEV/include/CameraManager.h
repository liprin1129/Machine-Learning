#ifndef __CAMERA_MANAGER_H__
#define __CAMERA_MANAGER_H__

#include <sl_zed/Camera.hpp>
#include <opencv2/opencv.hpp>

class CameraManager {
    private:
        // Instance variables
        sl::Camera _zed;
        sl::InitParameters _init_params;
        sl::RuntimeParameters _runtime_params;

        int imgViewWidth;
        int imgViewHeight;
        
        // Intrinsic
        int focalLength;
        int fx;
        int fy;

        sl::Mat slMat;
        cv::Mat cvMat;

        // Methods
        void initParams(); // Initial camera configs
        void runtimeParams(); // Runtime camera configs
        int cameraOpen();

        cv::Mat getOneFrame();
    public:
        // Getter
        int getFocalLength(){return focalLength;};
        
        // Constructor function for CameraManager class
        CameraManager();
        ~CameraManager();
        std::vector<double> getJointZ();

        // main method for CameraManger class
        void cameraManagerDidLoad(int argc, char **argv);
};

#endif