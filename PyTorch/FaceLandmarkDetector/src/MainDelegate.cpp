#include "MainDelegate.h"

/*
void checkTensorImgAndLandmarksV1(torch::Tensor const &imgTensor, torch::Tensor const &labelTensor) {
    // Convert the image Tensor to cv::Mat with CV_8UC3 data type
    int cvMatSize[2] = {(int)imgTensor.size(1), (int)imgTensor.size(2)};
    cv::Mat imgCV(2, cvMatSize, CV_32FC3, imgTensor.data_ptr());
    imgCV.convertTo(imgCV, CV_8UC3);

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
*/
int MainDelegate::mainDelegation(int argc, char** argv){
   // Create the device we pass around based on whether CUDA is available.
   //if (torch::cuda::is_available()) {

   // Data Loader
   CustomDataset dl("/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv");//("/DATASETs/Face/Landmarks/300W/");
   for (int i=0; i<5; ++i){
      auto sample = dl.get(i);
      //std::cout << sample.data.sizes() << ", " << sample.target.sizes() <<  std::endl;
      //checkTensorImgAndLandmarksV1(sample.data, sample.target);
   }
   //std::cout << dl.size() << std::endl;

   /*FaceLandmarkNet fln(false, false); // verbose, test

   // Optimizer
   torch::optim::Adam adamOptimizer(
      fln->parameters(),
      torch::optim::AdamOptions(1e-4).beta1(0.5));

   std::cout << "CUDA is available! Training on GPU." << std::endl;
   fln->train(dl, torch::Device(torch::kCUDA), adamOptimizer);
   */
}