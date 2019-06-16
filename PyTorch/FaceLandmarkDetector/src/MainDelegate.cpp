#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
   /*auto inX = torch::randn({1, 3, 1004, 753});
   //std::cout << inX.sizes() << std::endl;

   FaceLandmarkDetector fld(true);
   std::cout << fld.getConv() << std::endl;
   fld.forward(inX);
   */
   
   DataLoader dl("/DATASETs/Face/Landmarks/300W/");
   dl.readDataAndLabels();
   //dl.readImage2CVMat(std::get<0>(dl.getDataset()[1]));
   dl.labelStr2Float(dl.getDataset()[0], true);
   
   for (auto &l: dl.getLabels()) std::cout << l << std::endl;
   cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
   cv::imshow("Image", dl.getImage());
   cv::waitKey(0);
   
   //dl.labelNormalizer(std::get<1>(dl.getDataset()[0]));
}