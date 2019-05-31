#include "CameraManager.h"

void CameraManager::initParams(){
    _init_params.camera_resolution = sl::RESOLUTION_HD1080;
    _init_params.depth_mode = sl::DEPTH_MODE_NONE;
    _init_params.coordinate_units = sl::UNIT_METER;
}