#include <sl_zed/Camera.hpp>
#include <opencv2/opencv.hpp>

class CameraManager {
    private:
        // Instance variables
        sl::Camera _zed;
        sl::InitParameters _init_params;
        sl::RuntimeParameters _runtime_params;

        sl::Mat slMat;
        cv::Mat cvMat;

        // Methods
        void initParams(); // Initial camera configs
        void runtimeParams(); // Runtime camera configs
        int cameraOpen();

        sl::Mat slMat2cvMatBridge(); // Pointer for slMat to cvMat
        cv::Mat slMat2cvMatConverter(sl::Mat& input);
    public:
        // Constructor function for CameraManager class
        CameraManager();

        // main method for CameraManger class
        int cameraManagerDidLoad();
};