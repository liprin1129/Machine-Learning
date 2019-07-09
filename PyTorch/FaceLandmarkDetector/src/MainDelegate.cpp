#include "MainDelegate.h"

//#include <thread>         // std::this_thread::sleep_for
//#include <chrono>         // std::chrono::seconds

void testShow(int count, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor) {
    //std::cout << imgTensor.sizes() << std::endl;
    //std::cout << labelTensor.sizes() << std::endl<<std::endl;
    
    // Convert the image Tensor to cv::Mat with CV_8UC3 data type
    
    auto copiedImgTensor = imgTensor[0].toType(torch::kUInt8).clone();
    //auto copiedImgTensor = imgTensor[0].clone();
    //std::cout << "Passed\n";
    //std::fprintf(stdout, "TENSOR Sum: %d\n", copiedImgTensor.sum().item<int>());
    //std::cout << " Copied Tensor: " << copiedImgTensor.sum() << std::endl;

    //std::cout << copiedImgTensor.sizes() << std::endl;
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

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;

    auto copiedLabelTensor = labelTensor.clone();
    //std::cout << copiedLabelTensor.sizes() << std::endl;

    for (int i=0; i<copiedLabelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = copiedLabelTensor[0][i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            if (Y < imgCV.rows and X < imgCV.cols) {
                cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 1, cv::Scalar( 54, 54, 251 ), cv::FILLED, cv::LINE_4);
            }
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = copiedLabelTensor[0][i].item<float>();//*outputImg.cols;
    }
    
    //cv::namedWindow("Restored", CV_WINDOW_NORMAL);
    //imshow("Restored", imgCV);
    //cv::waitKey(0);

    char outputString[100];
    std::sprintf(outputString, "./checkpoints/Images/test-output-%03d.jpg", count);
    cv::imwrite( outputString, imgCV );
}

void testSave(int count, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor) {
    //std::cout << imgTensor.sizes() << std::endl;
    //std::cout << labelTensor.sizes() << std::endl<<std::endl;
    
    // Convert the image Tensor to cv::Mat with CV_8UC3 data type
    
    auto copiedImgTensor = imgTensor[0].toType(torch::kUInt8).clone();
    //auto copiedImgTensor = imgTensor[0].clone();
    //std::cout << "Passed\n";
    //std::fprintf(stdout, "TENSOR Sum: %d\n", copiedImgTensor.sum().item<int>());
    //std::cout << " Copied Tensor: " << copiedImgTensor.sum() << std::endl;

    //std::cout << copiedImgTensor.sizes() << std::endl;
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

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;

    auto copiedLabelTensor = labelTensor.clone();
    //std::cout << copiedLabelTensor.sizes() << std::endl;

    for (int i=0; i<copiedLabelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = copiedLabelTensor[0][i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            if (Y < imgCV.rows and X < imgCV.cols) {
                cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 1, cv::Scalar( 54, 54, 251 ), cv::FILLED, cv::LINE_4);
            }
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = copiedLabelTensor[0][i].item<float>();//*outputImg.cols;
    }

    char outputString[100];
    std::sprintf(outputString, "./checkpoints/Images/test-output-%03d.jpg", count);
    cv::imwrite( outputString, imgCV );
}

int MainDelegate::mainDelegation(int argc, char** argv){
   // Create the device we pass around based on whether CUDA is available.
   if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;

        //TrainerInferrer ti(5000, 10, std::make_tuple(300, 300), false);
        //ti.train(
            //torch::Device(torch::kCUDA), 
            //"DATASETs/Face/Landmarks/300W-Dataset/300W/Data/", 
            //"DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv");

        //ti.infer(torch::Device(torch::kCUDA), 
        //    //"/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/indoor_001.png",
        //    "croped-left.png",
        //    "./checkpoints/Trained-models/output-epoch2600.pt");

        /*
        auto cds = CustomDataset(
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv",
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/",
            //"/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv",
            //"/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/",
            false)
            .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::Resize(500)))
            .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::RandomContrastBrightness(0.7, 3.0, 50.0)))
            .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::RandomCrop(0.7, 100.0)))
            .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::MiniMaxNormalize()))
            .map(torch::data::transforms::Stack<>());

        
        // Generate a data loader.
        //auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(cds),
            torch::data::DataLoaderOptions().batch_size(1).workers(1));
        
        int count = 0;
        for (auto& batch : *data_loader) {
            
            std::cout << batch.data.sizes() << batch.target.sizes() 
                << batch.data.min().item<float>() << ", \t" << batch.data.max().item<float>() << "\t | "
                << batch.target.min().item<float>() << ", " << batch.target.max().item<float>() << std::endl;

            testShow(++count, batch.data*255, batch.target*batch.data.size(2));
            //std::cout << count << std::endl;
            if (count > 10) {break;}
        }
        */

        train(
            false,
            torch::Device(torch::kCUDA),
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/",
            "/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv",
            1e-3,   // learning rate
            1000,   // epoch
            10,    // batch, 10
            6,      // workers
            0.5,    // data wrangling probability
            128,    // resize output
            3,      // contrast (alpha)
            50,     // brightness (beta)
            10     // move in pixel
        );    
    }
}

void MainDelegate::train(
    bool verbose, torch::Device device, std::string imgFolderPath, std::string labelCsvFile, 
    float learningRate, int numEpoch, int numMiniBatch, int numWorkers,
    float wranglingProb, int resizeFactor, float contrastFactor, float brightnessFactor, float moveFactor)
{    
    FaceLandmarkNet fln(3, verbose); // Num of channels, Verbose
    
    fln->to(device); // Upload the model to the device {CPU, GPU}

    // Optimizer
    torch::optim::Adam adamOptimizer(fln->parameters(), torch::optim::AdamOptions(learningRate).beta1(0.5));

    auto cds = CustomDataset(
        labelCsvFile,
        imgFolderPath,
        //"/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv",
        //"/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/",
        false)
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::Resize(resizeFactor))) // 300 or 500
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::RandomContrastBrightness(wranglingProb, contrastFactor, brightnessFactor))) // 0.7, 3.0, 50.0
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::RandomCrop(wranglingProb, moveFactor))) // 0.7, 100.0
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::MiniMaxNormalize()))
        .map(torch::data::transforms::Stack<>());

    // Generate a data loader.
    //auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(cds), 
        torch::data::DataLoaderOptions().batch_size(numMiniBatch).workers(numWorkers));


    for (int epoch = 0; epoch < numEpoch; ++epoch) 
    {
        float totLoss = 0;
        int batchCount = 0;

        for (auto& batch : *dataLoader) 
        {
            //auto data = batch.data;
            //auto labels = batch.target;

            if (not(torch::isnan(batch.data).sum().item<int>() or torch::isnan(batch.target).sum().item<int>())){
                
                adamOptimizer.zero_grad();
                torch::Tensor output = fln->forward(batch.data.to(device));

                torch::Tensor miniBatchLoss = torch::mse_loss(output, batch.target.to(device), Reduction::Mean);
                miniBatchLoss.backward(); // Calculate partial derivatives
                adamOptimizer.step(); // Back-propagation

                totLoss += miniBatchLoss.item<float>();
                //std::fprintf(stdout, "Batch Loss: %f, Total Loss: %f\n", miniBatchLoss.item<float>(), totLoss);
                ++batchCount;
                if ((epoch+1)%3 == 0 and batchCount == 1) {
                    testShow(++epoch, batch.data*255, output*batch.data.size(2));
                }
            }
        }

        // Test
        std::fprintf(stdout, "Epoch #[%d/%d] | (Train total loss: %f)\n", epoch+1, numEpoch, (totLoss*sqrt(pow(resizeFactor, 2) + pow(resizeFactor, 2)))/batchCount);
        //batchCount = 0;
        /*
        if ((epoch+1) % 3 == 0) {
            cv::Mat imgCV = cv::imread("/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/indoor_001.png", CV_LOAD_IMAGE_COLOR);
            imgCV.convertTo(imgCV, CV_32FC3);
            cv::resize(imgCV, imgCV, cv::Size2d(resizeFactor, resizeFactor), 0, 0, cv::INTER_LINEAR);
            torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
            torch::Tensor testTensor = torch::from_blob(imgCV.data, {imgCV.rows, imgCV.cols, 3}, imgOptions);
            testTensor = testTensor.permute({2, 0, 1}); // convert to CxHxW
            std::vector<torch::Tensor> testTensorVec;
            testTensorVec.reserve(2);
            testTensorVec.push_back(testTensor);
            testTensorVec.push_back(testTensor);

            torch::Tensor stackedTensor = torch::stack(testTensorVec);
            //testTensor = testTensor.unsqueeze(0);
            stackedTensor /= 255; // Normalization

            //std::cout << stackedTensor.sizes() << std::endl;
            torch::Tensor output = fln->forward(stackedTensor.to(device));

            testSave(epoch+1, stackedTensor*255, output*stackedTensor.size(2));
        }*/
    }
}

void MainDelegate::infer(torch::Device device, std::string imgPath, std::string modelPath) {

}