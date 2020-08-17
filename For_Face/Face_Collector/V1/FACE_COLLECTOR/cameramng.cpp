#include "cameramng.h"

CameraMng::CameraMng()
{
    initParams.camera_resolution = sl::RESOLUTION::HD720;
    //initParams.depth_mode = sl::DEPTH_MODE::NONE;
    initParams.coordinate_units = sl::UNIT::METER;

    //runParams.sensing_mode = sl::SENSING_MODE::STANDARD;
    //runParams.enable_depth = false;
    //runParams.measure3D_reference_frame = sl::REFERENCE_FRAME::CAMERA;

    //bool isOpened;
    //auto [isOpened, errMsg] = openCamera();
}

CameraMng::~CameraMng()
{
    cvTmpGpuMatL.release();
    cvTmpGpuMatR.release();
    cvGpuMatL.release();
    cvGpuMatR.release();

    zedGpuMatL.free();
    zedGpuMatR.free();

    zed.close();
    std::cout << "Camera closed." << std::endl;
}


std::tuple<bool, std::string> CameraMng::openCamera()
{
    //bool errCode = true;
    std::string errMsg = "Camera has opened.";
    // Open the camera
    sl::ERROR_CODE err = this->zed.open(initParams);
    if (err != sl::ERROR_CODE::SUCCESS) {
        //std::cout << toString(err) << std::endl;
        //this->zed.close();

        //assert(assertCode!=0); // Quit if an error occurred
        //errCode = false;
        errMsg = "Can't open the camera with error.";
        /*if (errCode == true) {
            return {errCode, errMsg};
        }*/
    }

    //isOpened = zed.isOpened();
    return {zed.isOpened(), errMsg};
}

bool CameraMng::getOneFrameFromZED()
{
    if (zed.grab(runParams) == sl::ERROR_CODE::SUCCESS) {
        // Retrieve the left image, depth image in half-resolution
        zed.retrieveImage(zedGpuMatL, sl::VIEW::LEFT, sl::MEM::GPU);
        zed.retrieveImage(zedGpuMatR, sl::VIEW::RIGHT, sl::MEM::GPU);
        //zed.retrieveImage(zedGpuMatL, sl::VIEW::LEFT, sl::MEM::GPU, sl::Resolution(640, 360));
        //zed.retrieveImage(zedGpuMatR, sl::VIEW::RIGHT, sl::MEM::GPU, sl::Resolution(640, 360));

        cvGpuMatL = slGpuMat2cvGpuMat(zedGpuMatL);
        cvGpuMatR = slGpuMat2cvGpuMat(zedGpuMatR);
        //std::cout << "1:   " << zedGpuMatL.getDataType() << std::endl;
        //std::cout << "2:   " << cvGpuMatL.type() << std::endl;

        cv::cuda::cvtColor(cvGpuMatL, cvTmpGpuMatL, cv::COLOR_BGRA2RGB);
        cv::cuda::cvtColor(cvGpuMatR, cvTmpGpuMatR, cv::COLOR_BGRA2RGB);

        cv::cuda::flip(cvTmpGpuMatL, cvGpuMatL, 1);
        cv::cuda::flip(cvTmpGpuMatR, cvGpuMatR, 1);
        //cvGpuMatL.download(cvCpuMatL);
        //cvGpuMatR.download(cvCpuMatR);

        //std::cout << "3:   " << cvGpuMatL.type() << std::endl;
        //std::cout << "4:   " << cvCpuMatL.type() << std::endl;

        return true;
    }

    return false;
}


cv::Mat CameraMng::slGpuMat2cvMat(sl::Mat &slMat)
{
    std::cout << "GPU SL Mat" << std::endl;

    if (slMat.getMemoryType() == sl::MEM::GPU) {
        std::cout << "GPU SL Mat" << std::endl;
        slMat.updateCPUfromGPU();
    }

    int cvType = -1;
    switch (slMat.getDataType()){
        case sl::MAT_TYPE::F32_C1: cvType = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cvType = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cvType = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cvType = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cvType = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cvType = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cvType = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cvType = CV_8UC4; break;
        default: break;
    }

    return cv::Mat(static_cast<int>(slMat.getHeight()),
            static_cast<int>(slMat.getWidth()),
            cvType,
            slMat.getPtr<sl::uchar1>(sl::MEM::CPU),
            slMat.getStepBytes(sl::MEM::CPU));
}


cv::cuda::GpuMat CameraMng::slGpuMat2cvGpuMat(sl::Mat &slMat)
{
    //std::cout << "GPU SL Mat" << std::endl;

    /*if (slMat.getMemoryType() == sl::MEM_GPU) {
        //std::cout << "GPU SL Mat" << std::endl;
        slMat.updateCPUfromGPU();
    }*/

    int cvType = -1;
    switch (slMat.getDataType()){
        case sl::MAT_TYPE::F32_C1: cvType = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cvType = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cvType = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cvType = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cvType = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cvType = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cvType = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cvType = CV_8UC4; break;
        default: break;
    }

    return cv::cuda::GpuMat(
                static_cast<int>(slMat.getHeight()), static_cast<int>(slMat.getWidth()),
                cvType,
                slMat.getPtr<sl::uchar1>(sl::MEM::GPU), slMat.getStepBytes(sl::MEM::GPU));
}

std::tuple<cv::Mat, cv::Mat> CameraMng::getCpuMat()
{
    cv::Mat cvCpuMatL, cvCpuMatR;

    cvGpuMatL.download(cvCpuMatL);
    cvGpuMatR.download(cvCpuMatR);

    return {cvCpuMatL, cvCpuMatR};
}

std::tuple<cv::Mat, cv::Mat> CameraMng::cvtGpuMat2CpuMat(cv::cuda::GpuMat cvGpuMatL, cv::cuda::GpuMat cvGpuMatR)
{
    cv::Mat cvCpuMatL, cvCpuMatR;

    cvGpuMatL.download(cvCpuMatL);
    cvGpuMatR.download(cvCpuMatR);

    return {cvCpuMatL, cvCpuMatR};
}

std::tuple<cv::cuda::GpuMat, cv::cuda::GpuMat> CameraMng::getGpuMat()
{
    return {cvGpuMatL, cvGpuMatR};
}
