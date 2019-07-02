#include "MainDelegate.h"

void checkTensorImgAndLandmarks(torch::Tensor const &imgTensor, torch::Tensor const &labelTensor) {
    // Convert the image Tensor to cv::Mat with CV_8UC3 data type
    auto copiedImgTensor = imgTensor.toType(torch::kUInt8).clone();

    int cvMatSize[2] = {(int)imgTensor.size(1), (int)imgTensor.size(2)};
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

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;
    for (int i=0; i<labelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = labelTensor[0][i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 2, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = labelTensor[0][i].item<float>();//*outputImg.cols;
    }
    
    cv::namedWindow("Restored", CV_WINDOW_AUTOSIZE);
    imshow("Restored", imgCV);
    cv::waitKey(0);
}


int MainDelegate::mainDelegation(int argc, char** argv){
   // Create the device we pass around based on whether CUDA is available.
   if (torch::cuda::is_available()) {
        FaceLandmarkNet fln(1000, 100, std::make_tuple(300, 300), false, false); // verbose, test

        // Optimizer
        torch::optim::Adam adamOptimizer(
            fln->parameters(),
            torch::optim::AdamOptions(1e-4).beta1(0.5));

        std::cout << "CUDA is available! Training on GPU." << std::endl;
        fln->train(torch::Device(torch::kCUDA), adamOptimizer);

   // Data Loader
   //CustomDataset dl(
   //    "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv", 
   //    "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/");//("/DATASETs/Face/Landmarks/300W/");

    //auto rescale = Rescale(std::make_tuple(200, 200));

    /* // torch data loader test
    //torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>> 
    auto cds = CustomDataset(
        "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv", 
        "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/",
        std::make_tuple(200, 200))
        .map(torch::data::transforms::Normalize<>(255, 255))
        .map(torch::data::transforms::Stack<>());

    // Generate a data loader.
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(cds), 
        torch::data::DataLoaderOptions().batch_size(10).workers(8));

    // In a for loop you can now use your data.
    for (auto& batch : *data_loader) {
        auto data = batch.data;
        auto labels = batch.target;
        // do your usual stuff
        std::cout << data.sizes() << std::endl;
        std::cout << labels.sizes() << std::endl;

        for (int i=0; i<9; ++i)
            std::cout << data[i].min().item<float>() << ", " << data[i].max().item<float>() << std::endl;
            //std::cout << labels[i].min().item<float>() << ", " << labels[i].max().item<float>() << std::endl;
            //std::cout << labels[i] << std::endl;
        std::cout << std::endl;
    }
    */

    /*
    CustomDataset dl(
        "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv", 
        "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/",
        std::make_tuple(200, 200));
        
   for (int i=10; i<50; ++i){
      auto sample = dl.get(i);
      //std::cout << sample.data.sizes() << ", " << sample.target.sizes() <<  std::endl;
      checkTensorImgAndLandmarksV1(sample.data, sample.target);
   }*/
   //std::cout << dl.size() << std::endl;
    
    /*// Rescale Class test
    std::tuple<int, int> testSize = {10, 50};
    auto rescale = Rescale(testSize);
    cv::Mat testA = cv::Mat::ones(200, 100, CV_8UC1);
    std::vector<int> testB = {100, 200, 300};
    rescale(testA, testB);*/
    }
}