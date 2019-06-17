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
   
   double min, max;
   cv::minMaxIdx(dl.getImage(), &min, &max);
   std::fprintf(stdout, "Image Info: \n\tMin: %f | Max: %f \n", min, max);
   //for (auto &l: dl.getLabels()) std::fprintf(stdout, "%lf, ", l);

   float minLable=1.0, maxLabel=0.0;

   for (auto &l: dl.getLabels()) {
      if (l < minLable) {
         minLable = l;
      }

      if (l > maxLabel) {
         maxLabel = l;
      }
   }
   std::fprintf(stdout, "Labels Info: \n\tMin: %f | Max: %f\n", minLable, maxLabel);

   // Convert byte tensor to float
   std::vector<int64_t> sizes = {1, 3, dl.getImage().cols, dl.getImage().rows};
   at::TensorOptions options(at::kFloat);
   at::Tensor inX = torch::from_blob(dl.getImage().data, at::IntList(sizes), options);
   //inX = inX.toType(at::kFloat);

   std::cout << "\n inX Tensor Info.:" << std::endl;
   std::cout << "\t size: " << inX.sizes() << std::endl;
   std::cout << "\t max: " << inX.max() << std::endl;
   std::cout << "\t min: " << inX.min() << std::endl;
   
   //std::cout << inX << std::endl;

   torch::Tensor output = fln.forward(inX);

   
   double labelsArr[136];
   //std::copy(dl.getLabels().begin(), dl.getLabels().end(), labelsArr);
   int k = 0;
   for (auto &a: dl.getLabels()) {
      labelsArr[k++] = a;
   }
   //std::cout << labelsArr[0] << std::endl;
   auto labels = torch::tensor(
                           labelsArr,
                           torch::requires_grad(false).dtype(at::kFloat)).view({1, 136});
   
   std::cout << "\nlabels Tensor Info.:" << std::endl;
   std::cout << "\t size: " << labels.sizes() << std::endl;
   std::cout << "\t max: " << labels.max() << std::endl;
   std::cout << "\t min: " << labels.min() << std::endl;
   
   torch::Tensor loss = torch::mse_loss(output, labels);
   std::cout << "\nMSE: " << loss;
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