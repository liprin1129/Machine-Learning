#include <sl_zed/Camera.hpp>
#include <opencv2/opencv.hpp>

class CameraManager {
    private:
        // Instance variables
        sl::Camera _zed;
        sl::InitParameters _init_params;

        // Methods
        void initParams();
}