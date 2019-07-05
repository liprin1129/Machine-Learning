#include "MainDelegate.h"


int MainDelegate::mainDelegation(int argc, char** argv){
   // Create the device we pass around based on whether CUDA is available.
   if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;

        TrainerInferrer ti(5000, 50, std::make_tuple(300, 300), false);
        ti.train(
            torch::Device(torch::kCUDA), 
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/", 
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv");

        //ti.infer(torch::Device(torch::kCUDA), 
        //    //"/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/indoor_001.png",
        //    "croped-left.png",
        //    "./checkpoints/Trained-models/output-epoch2600.pt");
    }
}