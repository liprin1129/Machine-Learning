#include "facelandmarker.h"
#include <opencv2/opencv.hpp>

FaceLandmarker::FaceLandmarker()
{
    //std::cout << "FaceLandmarker constructor has loaded 1." << std::endl;
    // CascadeClassifier instance
    gpuCascade = cv::cuda::CascadeClassifier::create(
                "data/haarcascade_frontalface_alt.xml");
    //std::cout << "FaceLandmarker constructor has loaded 2." << std::endl;
    cudaCascadeParamSetup(1.5, false, 1, true);
    //std::cout << "FaceLandmarker constructor has loaded 3." << std::endl;

    faceL = cv::Rect(0, 0, 0, 0);
}

FaceLandmarker::~FaceLandmarker()
{
    gpuCascade.release();
}

void FaceLandmarker::cudaCascadeParamSetup(
        double scaleFactor, bool findLargestObject, int minNeighbors, bool filterRects)
{
    gpuCascade->setFindLargestObject(findLargestObject);
    gpuCascade->setScaleFactor(scaleFactor);
    gpuCascade->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
}

std::tuple<cv::Mat, cv::Mat> FaceLandmarker::findFaces(cv::cuda::GpuMat cvGpuMatL, cv::cuda::GpuMat cvGpuMatR)
{
    cv::cuda::GpuMat tmpGrayL, tmpGrayR, tmpObjL, tmpObjR;

    //std::cout << cvGpuMatL.size() << "\n";
    if ((!cvGpuMatL.empty()) | (!cvGpuMatR.empty()))
    {
        //qDebug() << "Start convert\n";
        //std::cout << cvGpuMatL.size() << "\n";
        //std::cout << "Face detection start." << std::endl;

        cv::cuda::cvtColor(cvGpuMatL, tmpGrayL, cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(cvGpuMatR, tmpGrayR, cv::COLOR_RGB2GRAY);
        //std::cout << tmpGrayL.size() << "\n";
        //qDebug() << "Convert passed\n";

        gpuCascade->detectMultiScale(tmpGrayL, tmpObjL);
        gpuCascade->detectMultiScale(tmpGrayR, tmpObjR);
        //qDebug() << "detect passed\n";

        gpuCascade->convert(tmpObjL, facesL);
        gpuCascade->convert(tmpObjR, facesR);
        //qDebug() << "face passed\n";

        //std::cout << facesL.size() << ", " << facesR.size() << "\n";

        // Draw rectangle
        this->drawRectOnFaces(cvGpuMatL, cvGpuMatR, 2);

        return {cvCpuMatL, cvCpuMatR};
    }
}

void FaceLandmarker::drawRectOnFaces(cv::cuda::GpuMat cvGpuMatL, cv::cuda::GpuMat cvGpuMatR, int thickness)
{
    cvGpuMatL.download(cvCpuMatL);
    cvGpuMatR.download(cvCpuMatR);
    faceL = cv::Rect();

    if (facesL.size() != 0){
        for(unsigned int i = 0; i < facesL.size(); ++i){
           cv::rectangle(cvCpuMatL, facesL[i], cv::Scalar(224, 66, 66), thickness);
           if (i >= 1) {
               break;
           }
        }
        //faceLocX = facesL[0].x;
        //faceLocY = facesL[0].y;
        faceL = facesL[0];
    }

    if (facesR.size() != 0){
        for(unsigned int i = 0; i < facesR.size(); ++i)
        {
           cv::rectangle(cvCpuMatR, facesR[i], cv::Scalar(224, 66, 66), thickness);

           if (i >= 1) {
                break;
           }
        }
    }
}
