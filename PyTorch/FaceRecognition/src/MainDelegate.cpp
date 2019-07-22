#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
    // For monocular image
    // To run this code:
    // ./FaceRecognizer /DEVs/Machine-Learning/PyTorch/FaceLandmarkDetector/build/checkpoints/Trained-models/backup-models/model-2000.pt /DATASETs/Face/Face-SJC/Original-Data/155/1.png
    FaceRecognizer fr;
    FaceLandmarkNet fln(3, false);
    torch::load(fln, argv[1]);

    std::string csvFilePath = "/DATASETs/Face/Face-SJC/Temp-Detection-Check-Data/face-master.csv";
    auto mvMaster = fr.csvReader(csvFilePath); // Mean and Variance master

    torch::Tensor imgTensor = dataWrangling::Utilities::cvImageToTensorConverter(std::string(argv[2]), 128);
    //std::cout << imgTensor.sizes() << std::endl;

    fr.faceRecognition(fln, imgTensor, mvMaster, torch::Device(torch::kCUDA));
}
