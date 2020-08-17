#ifndef HEADINGGAME_H
#define HEADINGGAME_H

#include <opencv2/opencv.hpp>
#include <QThread>

class HeadingGame : public QThread
{
private:
    int ballLocX, ballLocY;
    cv::Mat ballImg;
    bool headingSuccess;

    cv::Rect fL;
    cv::Mat bImg;

public:
    HeadingGame();
    void blendingTwoImages(cv::Mat baseImg);
    void calNewXYLoc(int baseImgWidth, int baseImgHeight);
    void headingSuccessCecker(cv::Rect faceL, cv::Mat baseImg);
    void setThreadingArgs(cv::Rect faceL, cv::Mat baseImg);
    void run();
};

#endif // HEADINGGAME_H
