#include "MainDelegate.h"

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

void testShow(torch::Tensor const &imgTensor, torch::Tensor const &labelTensor) {
    //std::cout << imgTensor.sizes() << std::endl;
    //std::cout << labelTensor.sizes() << std::endl<<std::endl;
    
    // Convert the image Tensor to cv::Mat with CV_8UC3 data type
    
    auto copiedImgTensor = imgTensor[0].toType(torch::kUInt8).clone();
    //std::cout << copiedImgTensor << std::endl;

    std::cout << "testShow TEST: 1\n";
    std::cout << copiedImgTensor.sizes() << std::endl;
    int cvMatSize[2] = {(int)copiedImgTensor.size(1), (int)copiedImgTensor.size(2)};
    cv::Mat imgCVB(2, cvMatSize, CV_8UC1, copiedImgTensor[0].data_ptr());
    cv::Mat imgCVG(2, cvMatSize, CV_8UC1, copiedImgTensor[1].data_ptr());
    cv::Mat imgCVR(2, cvMatSize, CV_8UC1, copiedImgTensor[2].data_ptr()); //CV_8UC1
    
    // Merge each channel to create colour cv::Mat
    cv::Mat imgCV; // Merged output cv::Mat
    std::vector<cv::Mat> channels;
    channels.push_back(imgCVB);
    channels.push_back(imgCVG);
    channels.push_back(imgCVR);
    cv::merge(channels, imgCV);
    //imgCV.convertTo(imgCV, CV_8UC3);
    std::cout << "testShow TEST: 2\n";
    std::cout << imgCV.size << std::endl;

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;

    auto copiedLabelTensor = labelTensor.clone();
    std::cout << "testShow TEST: 3\n";
    std::cout << copiedLabelTensor.sizes() << std::endl;

    for (int i=0; i<copiedLabelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = copiedLabelTensor[0][i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 2, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = copiedLabelTensor[0][i].item<float>();//*outputImg.cols;
    }
    std::cout << "testShow TEST: 4\n";
    
    //std::this_thread::sleep_for (std::chrono::seconds(5));
    cv::namedWindow("Restored", CV_WINDOW_AUTOSIZE);
    imshow("Restored", imgCV);
    cv::waitKey(0);
}

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


        auto cds = CustomDataset(
            //"/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv",
            //"/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/",
            "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv",
            "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/",
            false)
            .map(MyResize<torch::Tensor>(500))
            .map(torch::data::transforms::Stack<>());

        std::cout << "MainDelegate TEST 1\n";
        // Generate a data loader.
        //auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(cds), 
            torch::data::DataLoaderOptions().batch_size(10).workers(1));
        std::cout << "MainDelegate TEST 2\n";
        
        for (auto& batch : *data_loader) {
            std::cout << "MainDelegate TEST 3\n";
            auto data = batch.data;
            std::cout << data.sizes() << std::endl;
            std::cout << "MainDelegate TEST 4\n";
            auto target = batch.target;
            std::cout << target.sizes() << std::endl;
            //std::cout << target << std::endl; 
            std::cout << "MainDelegate TEST 5\n";

            testShow(batch.data, batch.target);

            break;
        }
    }
}