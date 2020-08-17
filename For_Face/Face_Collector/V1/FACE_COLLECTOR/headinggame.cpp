#include "headinggame.h"

HeadingGame::HeadingGame()
{
    //auto ballImg =
    //cv::resize(cv::imread("data/soccer_ball_transparent.png", cv::IMREAD_UNCHANGED), ballImg, cv::Size(), 0.2, 0.2);
    cv::resize(cv::imread("data/soccer_ball_transparent.png", cv::IMREAD_COLOR), ballImg, cv::Size(), 0.2, 0.2);
    //std::cout << "Ball: " << ballImg.size() << std::endl;

    headingSuccess = true;
}

void HeadingGame::calNewXYLoc(int baseImgWidth, int baseImgHeight)
{
    cv::Mat blendedImg;

    //std::cout << rand()%baseImg.size().width << std::endl;
    ballLocX = rand()%baseImgWidth;
    ballLocY = rand()%baseImgHeight;
    //std::cout << "1: " << ballLocX << ", " << ballLocY << std::endl;
    if (ballLocX >= baseImgWidth-ballImg.cols) {
        ballLocX -= ballImg.cols;
    }

    if (ballLocY >= baseImgHeight-ballImg.rows) {
        ballLocY -= ballImg.rows;
    }
    //std::cout << "2: " << ballLocX << ", " << ballLocY << std::endl;
    //ballLocX = rand()%baseImg.size().width - ballImg.size().width; // or ballImg.cols
    //ballLocY = rand()%baseImg.size().height- ballImg.size().height; // or ballImg.rows

    headingSuccess = false;
}

void HeadingGame::blendingTwoImages(cv::Mat baseImg)
{
    //std::cout << headingSuccess << std::endl;

    if (headingSuccess == true) {
        calNewXYLoc(baseImg.cols, baseImg.rows);
    }

    cv::Rect roi = cv::Rect(ballLocX, ballLocY, ballImg.cols, ballImg.rows);
    //std::cout << "roi: " << roi << std::endl;
    cv::addWeighted(baseImg(roi), 1.0, ballImg, 1.0, 0.0, baseImg(roi));
    //std::cout << "Blended: " << blendedImg.size() << std::endl;
}

void HeadingGame::headingSuccessCecker(cv::Rect faceL, cv::Mat baseImg)
{   if (!faceL.empty()) {
        if (
            (faceL.x+faceL.width/2 > ballLocX && faceL.x+faceL.width/2 < ballLocX + ballImg.cols) &&
                (faceL.y+faceL.height/2 > ballLocY && faceL.y+faceL.height/2 < ballLocY + ballImg.rows) )
        {
            //std::cout << "YES! \t" << faceL.x+faceL.width << ", " << faceL.y+faceL.height<< std::endl;
            //std::cout << "YES! \t" << ballLocX << ", " << ballLocY << std::endl;
            headingSuccess = true;
        }

        else {
            //std::cout << "No!" << std::endl;
            //headingSuccess = false;
        }
    }
    blendingTwoImages(baseImg);
}

void HeadingGame::setThreadingArgs(cv::Rect faceL, cv::Mat baseImg)
{
    this->fL = faceL;
    this->bImg = baseImg;
}

void HeadingGame::run()
{
    headingSuccessCecker(fL, bImg);
}
