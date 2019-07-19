#include "MainDelegate.h"

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds


int MainDelegate::mainDelegation(int argc, char** argv){
   // Create the device we pass around based on whether CUDA is available.
   if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        
        FaceLandmarkNet fln(3, false);                                      // Num of channels, Verbose
        auto ti = TrainerInferrer();

        /* // When training
        ti.train(
            fln,
            false,
            torch::Device(torch::kCUDA),
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/",
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv",
            1e-3,   // learning rate
            2000,   // epoch
            50,     // batch, 10
            14,      // workers
            0.8,    // data wrangling probability
            128,    // resize output
            3,      // contrast (alpha)
            100,     // brightness (beta)
            20,     // move in pixel
            20       // how much time to save
        );
        */

        
        // When inferring
        torch::load(fln, argv[1]);

        //Read images
        /*
        torch::Tensor leftImgTensor = dataWrangling::Utilities::cvImageToTensorConverter(std::string(argv[2]), 128);
        torch::Tensor rightImgTensor = dataWrangling::Utilities::cvImageToTensorConverter(std::string(argv[3]), 128);

        ti.infer(
            fln,                            // trained parameters location
            leftImgTensor,
            rightImgTensor,
            torch::Device(torch::kCUDA)     // computation device
        );
        */

        // To run this code: ./FaceLandmarkDetector ./checkpoints/Trained-models/backup-models/model-2000.pt /DATASETs/Face/Face-SJC/155
        std::vector<std::string> imgFiles = dataWrangling::Utilities::readFileNamesWithAbsPath(std::string(argv[2]));
        
        int count = 0;
        for (auto &file: imgFiles) {
            torch::Tensor imgTensor = dataWrangling::Utilities::cvImageToTensorConverter(file, 128);
            ti.inferMono(
                fln,
                imgTensor,
                torch::Device(torch::kCUDA),
                count++
            );
        }
    }
}
