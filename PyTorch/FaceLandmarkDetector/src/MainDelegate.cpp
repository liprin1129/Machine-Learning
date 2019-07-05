#include "MainDelegate.h"


int MainDelegate::mainDelegation(int argc, char** argv){
   // Create the device we pass around based on whether CUDA is available.
   if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;

        //TrainerInferrer ti(5000, 50, std::make_tuple(300, 300), false);
        //ti.train(
            //torch::Device(torch::kCUDA), 
            //"/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/", 
            //"/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv");

        //ti.infer(torch::Device(torch::kCUDA), 
        //    //"/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/indoor_001.png",
        //    "croped-left.png",
        //    "./checkpoints/Trained-models/output-epoch2600.pt");

        using InputBatch = std::vector<int>;
        using OutputBatch = std::string;
        auto cds = CustomDataset(
            //"/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv",
            //"/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/",
            "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv",
            "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/",
            std::make_tuple(300, 300),
            false)
            .map(MyResize<>());
            //.map(torch::data::transforms::Stack<>());

        // Generate a data loader.
        //auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(cds), 
            torch::data::DataLoaderOptions().batch_size(1).workers(1));


        for (auto& batch : *data_loader) {
            auto data = batch.data;
            auto labels = batch.target;

            std::cout << labels << std::endl;

            break;
        }
    }
}