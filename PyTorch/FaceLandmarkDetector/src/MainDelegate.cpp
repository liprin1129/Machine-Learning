#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
   //auto inX = torch::randn({1, 3, 1004, 753});
   //std::cout << inX.sizes() << std::endl;

   FaceLandmarkNet fln(true);
   //std::cout << fln.getConv() << std::endl;
   //fln.forward(inX);

   torch::optim::Adam adamOptimizer(
      fln.parameters(), 
      torch::optim::AdamOptions(1e-3).beta1(0.5));

   DataLoader dl("/DATASETs/Face/Landmarks/300W/");
   dl.loadOneTraninImageAndLabel(dl.getDataset()[0]);
   
   
   //torch::Tensor inX = torch::from_blob(dl.getImage().data, {1, 3, dl.getImage().cols, dl.getImage().rows}, at::ScalarType::Byte);
   torch::Tensor inX = torch::from_blob(dl.getImage().data, {1, 3, dl.getImage().cols, dl.getImage().rows});
   std::cout << inX.sizes() << std::endl;
   torch::Tensor output = fln.forward(inX);

   /*
   double labelsArr[136];
   std::copy(dl.getLabels().begin(), dl.getLabels().end(), labelsArr);

   auto labels = torch::tensor(
                           labelsArr,
                           torch::requires_grad(false).dtype(torch::kDouble));//.view({1, 3});

   std::cout << labels << std::endl;
   */

   /*
   DataLoader dl("/DATASETs/Face/Landmarks/300W/");
   //dl.readDataDirectory();
   //dl.readImage2CVMat(std::get<0>(dl.getDataset()[1]));
   dl.loadOneTraninImageAndLabel(dl.getDataset()[0]);
   
   double min, max;
   cv::minMaxIdx(dl.getImage(), &min, &max);
   std::fprintf(stdout, "Min: %lf | Max: %lf \n", min, max);
   for (auto &l: dl.getLabels()) std::fprintf(stdout, "%lf, ", l);
   std::fprintf(stdout, "\n");
   //cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
   //cv::imshow("Image", dl.getImage());
   //cv::waitKey(0);
   
   //dl.labelNormalizer(std::get<1>(dl.getDataset()[0]));
   */
}