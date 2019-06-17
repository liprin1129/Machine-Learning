#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){

    // Create the device we pass around based on whether CUDA is available.
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
    }

   //auto inX = torch::randn({1, 3, 1004, 753});
   //std::cout << inX.sizes() << std::endl;

   FaceLandmarkNet fln(false);
   fln.to(device);

   //std::cout << fln.getConv() << std::endl;
   //fln.forward(inX);

   // Optimizer
   torch::optim::Adam adamOptimizer(
      fln.parameters(), 
      torch::optim::AdamOptions(1e-3).beta1(0.5));

   // Data Loader
   DataLoader dl("/DATASETs/Face/Landmarks/300W/");
   
   /* // Check the loaded data is OK!
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
   */

   dl.loadOneTraninImageAndLabel(dl.getDataset()[0]);
   // Convert cv::Mat to Tensor
   std::vector<int64_t> sizes = {1, 3, dl.getImage().cols, dl.getImage().rows};
   at::TensorOptions options(at::kFloat);
   at::Tensor inX = torch::from_blob(dl.getImage().data, at::IntList(sizes), options).to(device);
   
   // Convert labels to Tensor
   double labelsArr[136];
   //std::copy(dl.getLabels().begin(), dl.getLabels().end(), labelsArr);
   int k = 0;
   for (auto &a: dl.getLabels()) {
      labelsArr[k++] = a;
   }
   
   torch::Tensor labels = torch::tensor(
                           labelsArr,
                           torch::requires_grad(false).dtype(at::kFloat)).view({1, 136}).to(device);

   for (int epoch = 0; epoch < 1000; ++epoch) {
      //torch::Tensor miniBatchLoss = torch::zeros(1, device);

      //for (auto &data: dl.getDataset()) {
         // Image and Label iterator

         adamOptimizer.zero_grad();
         torch::Tensor output = fln.forward(inX);
         
         /* // Converted image to tensor information
         std::cout << "\n inX Tensor Info.:" << std::endl;
         std::cout << "\t size: " << inX.sizes() << std::endl;
         std::cout << "\t max: " << inX.max() << std::endl;
         std::cout << "\t min: " << inX.min() << std::endl;
         */

         /* // Converted lables to tensor information
         std::cout << "\nlabels Tensor Info.:" << std::endl;
         std::cout << "\t size: " << labels.sizes() << std::endl;
         std::cout << "\t max: " << labels.max() << std::endl;
         std::cout << "\t min: " << labels.min() << std::endl;
         */
         
         torch::Tensor loss = torch::mse_loss(output, labels);
         loss.backward();
         adamOptimizer.step();
         
         if (epoch % 50) {
            //std::fprintf(stdout, "Epoch #%d: Mini Batch #%d (loss: %f)\n", epoch, miniBatchCounter, loss.item<float>());
            std::fprintf(stdout, "Epoch #%d | loss: %f\n", epoch, loss.item<float>());
         }

         /*
         std::cout << "\nMSE: " << loss.item<float>();

         miniBatchLoss += loss;

         //std::fprintf(stdout, "Epoch #%d: Mini Batch #%d\n", epoch, miniBatchCounter);

         --miniBatchCounter;
         if (miniBatchCounter < 1) {
            std::fprintf(stdout, "Epoch #%d: Mini Batch #%d\n", epoch, miniBatchCounter);
            
            miniBatchLoss /= miniBatchSize;
            miniBatchLoss.backward(torch::nullopt, true, false);
            adamOptimizer.step();

            miniBatchCounter = miniBatchSize;
         }
         */
      //}
   }

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