#include "MainDelegate.h"


int MainDelegate::mainDelegation(int argc, char** argv){
   // Create the device we pass around based on whether CUDA is available.
   if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;

        TrainerInferrer ti(5000, 150, std::make_tuple(120, 120), false);
        ti.train(
            torch::Device(torch::kCUDA), 
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/", 
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv");

        //fln->train(torch::Device(torch::kCUDA), adamOptimizer);
        //fln->infer(torch::Device(torch::kCUDA), 
        //    "/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/indoor_001.png",
        //    "./checkpoints/Trained-models/output-epoch0999.pt");
    }
}