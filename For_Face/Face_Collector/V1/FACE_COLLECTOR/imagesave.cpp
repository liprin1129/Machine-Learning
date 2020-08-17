#include "imagesave.h"

ImageSave::ImageSave()
{
    //this->dirPath = dirPath;
    imgCount = 0;
}

void ImageSave::setArgs(std::string dirPath, cv::Mat img)
{
    this->dirPath = dirPath;
    this->img = img;
}

void ImageSave::saveImg()
{
    //std::cout << dirPath + "/" + std::to_string(imgCount++) << ".png" << std::endl;
    std::string imgName = dirPath + "/" + std::to_string(imgCount++) + ".png";
    //std::cout << imgName << ".png" << std::endl;
    //std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    if (!img.empty()) {
        cv::Mat cvtImg;
        cv::cvtColor(img, cvtImg, cv::COLOR_RGB2BGR);
        cv::imwrite(imgName, cvtImg);
    }
}

void ImageSave::run() {
    if (!dirPath.empty()) {
        //std::cout << dirPath << std::endl;
        saveImg();
    }
    else {
        //std::cout << "empty!" << std::endl;
    }
    QThread::msleep(500);
}
