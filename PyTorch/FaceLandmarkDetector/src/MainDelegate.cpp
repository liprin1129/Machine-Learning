#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
   // Create the device we pass around based on whether CUDA is available.
   //if (torch::cuda::is_available()) {

   // Data Loader
   DataLoader dl("/DATASETs/Face/Landmarks/300W/");

   FaceLandmarkNet fln(false, false); // verbose, test

   // Optimizer
   torch::optim::Adam adamOptimizer(
      fln->parameters(),
      torch::optim::AdamOptions(1e-4).beta1(0.5));

   std::cout << "CUDA is available! Training on GPU." << std::endl;
   fln->train(dl, torch::Device(torch::kCUDA), adamOptimizer);
}