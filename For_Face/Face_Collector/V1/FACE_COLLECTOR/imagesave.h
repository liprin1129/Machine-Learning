#ifndef IMAGESAVE_H
#define IMAGESAVE_H

#include <QThread>

# include <opencv2/opencv.hpp>

class ImageSave : public QThread
{
private:
    cv::Mat img;

    std::string dirPath;
    int imgCount;
public:
    ImageSave();
    void setArgs(std::string dirPath, cv::Mat img);
    void saveImg();
    void run();
};

#endif // IMAGESAVE_H
