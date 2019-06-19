#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
   //at::globalContext().setUserEnabledCuDNN(false); // Need to check
   
   // Create the device we pass around based on whether CUDA is available.
   torch::Device device(torch::kCPU);
   if (torch::cuda::is_available()) {
   std::cout << "CUDA is available! Training on GPU." << std::endl;
   device = torch::Device(torch::kCUDA);
   }

   //auto inX = torch::randn({1, 3, 1004, 753});
   //std::cout << inX.sizes() << std::endl;

   FaceLandmarkNet fln(false);
   fln->to(device);

   //std::cout << fln.getConv() << std::endl;
   //fln.forward(inX);

   // Optimizer
   torch::optim::Adam adamOptimizer(
      fln->parameters(), 
      torch::optim::AdamOptions(1e-3).beta1(0.5));

   // Data Loader
   DataLoader dl("/DATASETs/Face/Landmarks/300W/");

   // Convert cv::Mat to Tensor
   std::vector<int64_t> sizes = {1, 3, dl.getImage().cols, dl.getImage().rows};
   //torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
   /*torch::Tensor inX;/* = torch::from_blob(
                        dl.getImage().data, 
                        {1, 3, dl.getImage().cols, dl.getImage().rows}, 
                        imgOptions).to(device);*/
   
   // Convert labels to Tensor
   double labelsArr[136];
   /*/int k = 0;
   for (auto &a: dl.getLabels()) {
      labelsArr[k++] = a;
   }*/
   //torch::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
   /*torch::Tensor label;/* = torch::tensor(
                           labelsArr,
                           labelOptions).view({1, 136}).to(device);*/
   
   for (int epoch = 0; epoch < 10; ++epoch) {
      //torch::Tensor miniBatchLoss = torch::zeros(1, device);
      int count = 0;

      for (auto &data: dl.getDataset()) { // Image and Label iterator
         auto [cvImg, listLabel] = dl.loadOneTraninImageAndLabel(data, true);

         //inX = torch::from_blob(
         at::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
         at::Tensor inX = torch::from_blob(
                        cvImg.data, 
                        {1, 3, dl.getImage().cols, dl.getImage().rows}, 
                        imgOptions).to(device);

         at::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
         at::Tensor label = torch::tensor(
                        labelsArr,
                        labelOptions).view({1, 136}).to(device);

         /*
         // Check memory locations
         std::fprintf(stdout, "Memory location: \n\t%p | %p\n", &cvImg, &listLabel);
         // Check the loaded image data is OK!
         double min, max;
         cv::minMaxIdx(cvImg, &min, &max);
         std::fprintf(stdout, "Image Info: \n\tMin: %f | Max: %f \n", min, max);
         //for (auto &l: dl.getLabels()) std::fprintf(stdout, "%lf, ", l);

         // Check the loaded labels are OK!
         float minLable=1.0, maxLabel=0.0;
         for (auto &l: listLabel) {
            if (l < minLable) {
               minLable = l;
            }

            if (l > maxLabel) {
               maxLabel = l;
            }
         }
         std::fprintf(stdout, "Labels Info: \n\tMin: %f | Max: %f\n", minLable, maxLabel);

         // Converted image to tensor information
         std::cout << "inX Tensor Info.:" << std::endl;
         std::cout << "\t size: " << inX.sizes() << std::endl;
         std::cout << "\t max: " << inX.max() << std::endl;
         std::cout << "\t min: " << inX.min() << std::endl;
         

         // Converted lables to tensor information
         std::cout << "labels Tensor Info.:" << std::endl;
         std::cout << "\t size: " << label.sizes() << std::endl;
         std::cout << "\t max: " << label.max() << std::endl;
         std::cout << "\t min: " << label.min() << std::endl;
         std::cout << std::endl;
         */

         if ((cvImg.cols < 1100) and (cvImg).rows < 1100) {
            //adamOptimizer.zero_grad();
            torch::Tensor output = fln->forward(inX);
            
            torch::Tensor loss = torch::mse_loss(output, label);
            loss.backward();
            adamOptimizer.step();
            
            ++count;
            if (count % 100) {
               //std::fprintf(stdout, "Epoch #%d: Mini Batch #%d (loss: %f)\n", epoch, miniBatchCounter, loss.item<float>());
               std::fprintf(stdout, "Epoch #%d, totCount #%d | loss: %f\n", epoch, count, loss.item<float>());
               //std::fprintf(stdout, "totCount #%d\n", totCount);
            }
         }
      }
   }
}