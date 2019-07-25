#include "Trainer+Inferrer.h"

void TrainerInferrer::testShow(int count, int resizeFactor, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor) {
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
    // Resize Image
    cv::resize(imgCV, imgCV, cv::Size2d(resizeFactor, resizeFactor), 0, 0, cv::INTER_LINEAR);

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;

    auto copiedLabelTensor = (labelTensor*resizeFactor).clone();
    //copiedLabelTensor = copiedLabelTensor*resizeFactor;
    //std::cout << copiedLabelTensor.sizes() << std::endl;

    for (int i=0; i<copiedLabelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = copiedLabelTensor[0][i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            if (Y < imgCV.rows and X < imgCV.cols) {
                cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 3, cv::Scalar( 54, 54, 251 ), cv::FILLED, cv::LINE_4);
            }
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = copiedLabelTensor[0][i].item<float>();//*outputImg.cols;
    }
    
    //cv::namedWindow("Restored", CV_WINDOW_NORMAL);
    //imshow("Restored", imgCV);
    //cv::waitKey(0);

    char outputString[100];
    std::sprintf(outputString, "./checkpoints/Images/TrainDataset/%03d.jpg", count);
    cv::imwrite( outputString, imgCV );
}

void TrainerInferrer::testSave(int count, int rescaleFactor, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor, char* outputName) {
    //std::cout << imgTensor.sizes() << std::endl;
    //std::cout << labelTensor.sizes() << std::endl<<std::endl;
    
    // Convert the image Tensor to cv::Mat with CV_8UC3 data type
    
    auto copiedImgTensor = imgTensor.toType(torch::kUInt8).clone();
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

    //Resize image
    cv::resize(imgCV, imgCV, cv::Size2d(rescaleFactor, rescaleFactor), 0, 0, cv::INTER_LINEAR);

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;

    auto copiedLabelTensor = (labelTensor*rescaleFactor).clone();
    // Resize labels
    //copiedImgTensor = copiedImgTensor*rescaleFactor;
    //std::cout << copiedLabelTensor.sizes() << std::endl;

    for (int i=0; i<copiedLabelTensor.size(0); ++i) {
        if (i % 2 == 1) {
            Y = copiedLabelTensor[i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            if (Y < imgCV.rows and X < imgCV.cols) {
                cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 3, cv::Scalar( 54, 54, 251 ), cv::FILLED, cv::LINE_4);
            }
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = copiedLabelTensor[i].item<float>();//*outputImg.cols;
    }

    char outputString[100];
    std::sprintf(outputString, "./checkpoints/Images/TestDataset/%s/%03d.jpg", outputName, count);
    cv::imwrite( outputString, imgCV );
}

void TrainerInferrer::writeCSV(int count, torch::Tensor const &labelTensor) {
    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    std::vector<float> xVec;
    std::vector<float> yVec;
    std::vector<float> xNormVec;
    std::vector<float> yNormVec;

    auto copiedLabelTensor = (labelTensor).clone();

    for (int i=0; i<copiedLabelTensor.size(0); ++i) {
        // For y coordinate
        if (i % 2 == 1) {
            if (copiedLabelTensor[i].item<float>() < 1e-4) { // if the value is less than 0.0001, let the value be 0.
                yVec.push_back(0);
            }
            else { // otherwise, round at 4 decimal points
                yVec.push_back(
                    round(copiedLabelTensor[i].item<float>() * 1e+4) / 1e+4
                );
            }
        }
        
        // For x coordinate
        else {
            if (copiedLabelTensor[i].item<float>() < 1e-4) {
                    xVec.push_back(0);
            }
            else {
                xVec.push_back( 
                    round(copiedLabelTensor[i].item<float>() * 1e+4) / 1e+4
                );
            }
        }
    }

    //for (int i=0; i<copiedLabelTensor.size(0)/2; ++i)
    //    std::cout << i+1 << " : " << xVec[i] << ", " << yVec[i] << "\n";
    
    // Find maximum x and y
    auto xMax = std::max_element(xVec.begin(), xVec.end());
    auto xMin = std::min_element(xVec.begin(), xVec.end()) + 1e-8;

    auto yMax = std::max_element(yVec.begin(), yVec.end());
    auto yMin = std::min_element(yVec.begin(), yVec.end()) + 1e-8;

    // Rescale to [-1, 1]
    std::for_each(xVec.begin(), xVec.end(), 
        [&](float &x){ 
            float normX = 2*( (x - *xMin) / (*xMax - *xMin) ) - 1 ;
            xNormVec.push_back( round(normX * 1e+4) / 1e+4 );
        });

    std::for_each(yVec.begin(), yVec.end(), 
        [&](float &y){ 
            float normY = 2*( (y - *yMin) / (*yMax - *yMin) ) - 1;
            yNormVec.push_back( round(normY * 1e+4) / 1e+4 );
        });
    
    /*// Rescale to [0, 1]
    std::for_each(xVec.begin(), xVec.end(), 
        [&](float &x){ 
            float normX = (x - *xMin) / (*xMax - *xMin) ;
            xNormVec.push_back( round(normX * 1e+4) / 1e+4 );
        });

    std::for_each(yVec.begin(), yVec.end(), 
        [&](float &y){ 
            float normY = (y - *yMin) / (*yMax - *yMin);
            yNormVec.push_back( round(normY * 1e+4) / 1e+4 );
        });
    */


    // Save to CSV
    std::fstream fout; 
    std::string outputString("./checkpoints/Images/TestDataset/output-landmarks.csv");
    // opens an existing csv file or creates a new file. 
    fout.open(outputString, std::ios::out | std::ios::app);

    for (int i=0; i<xNormVec.size(); ++i) {
        fout << xNormVec[i] << ", ";
        fout << yNormVec[i];

        if (i < xNormVec.size()-1) {
            fout << ", ";
        }
        
        else {
            fout << "\n";
        }
    }

    fout.clear();
}

//void TrainerInferrer::overlapOuputsOnCVMat(cv::Mat cvMat, const at::Tensor &labelTensor) {
//    dataWrangling::
//}

void TrainerInferrer::inferStereoToCSV(FaceLandmarkNet fln, const at::Tensor &leftImageTensor, const at::Tensor &rightImageTensor, torch::Device device) {    
    fln->eval();                        // evaluation mode
    fln->to(device);                    // Upload the model to the device {CPU, GPU}

    /*
    // Convert to Tensors
    torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor leftImageTensor = torch::from_blob(leftImgCV.data, {leftImgCV.rows, leftImgCV.cols, 3}, imgOptions);
    torch::Tensor rightImageTensor = torch::from_blob(rightImgCV.data, {rightImgCV.rows, rightImgCV.cols, 3}, imgOptions);
    
    leftImageTensor = leftImageTensor.permute({2, 0, 1}); // convert to CxHxW
    rightImageTensor = rightImageTensor.permute({2, 0, 1}); // convert to CxHxW
    */
    std::vector<torch::Tensor> testTensorVec;
    testTensorVec.reserve(2);
    testTensorVec.push_back(leftImageTensor);
    testTensorVec.push_back(rightImageTensor);
    torch::Tensor stackedTensor = torch::stack(testTensorVec);

    stackedTensor /= 255; // Normalization

    torch::Tensor output = fln->forward(stackedTensor.to(device)).detach();

    testSave(0, 500, stackedTensor[0]*255, output[0], (char*) "Left"); // count to save image, resize factor after the prediction, img tensor, output label tensor, characters indicating to folder name
    testSave(0, 500, stackedTensor[1]*255, output[1], (char*) "Right"); // count to save image, resize factor after the prediction, img tensor, output label tensor, characters indicating to folder name
}

void TrainerInferrer::inferMono(FaceLandmarkNet fln, const at::Tensor &imageTensor, torch::Device device, int imgCount) {
    fln->eval();                        // evaluation mode
    fln->to(device);                    // Upload the model to the device {CPU, GPU}

    /*
    // Convert to Tensors
    torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor leftImageTensor = torch::from_blob(leftImgCV.data, {leftImgCV.rows, leftImgCV.cols, 3}, imgOptions);
    torch::Tensor rightImageTensor = torch::from_blob(rightImgCV.data, {rightImgCV.rows, rightImgCV.cols, 3}, imgOptions);
    
    leftImageTensor = leftImageTensor.permute({2, 0, 1}); // convert to CxHxW
    rightImageTensor = rightImageTensor.permute({2, 0, 1}); // convert to CxHxW
    */
    std::vector<torch::Tensor> testTensorVec;
    testTensorVec.reserve(2);
    testTensorVec.push_back(imageTensor);
    testTensorVec.push_back(imageTensor);
    torch::Tensor stackedTensor = torch::stack(testTensorVec);

    stackedTensor /= 255; // Normalization

    torch::Tensor output = fln->forward(stackedTensor.to(device)).detach();

    writeCSV(imgCount, output[0]);
}

std::tuple<cv::Mat, cv::Mat> TrainerInferrer::inferStereo
(
    FaceLandmarkNet fln, 
    cv::Mat leftCVImg, cv::Mat rightCVImg, 
    //const at::Tensor &leftImageTensor, const at::Tensor &rightImageTensor, 
    torch::Device device
) 
{
    std::cout << "I'm in" << std::endl;
    if (leftCVImg.rows > 0 && leftCVImg.cols > 0 && rightCVImg.rows > 0 && rightCVImg.cols > 0)
    {
        auto leftImageTensor = dataWrangling::Utilities::cvImageToTensorConverter(leftCVImg, 128);
        auto rightImageTensor = dataWrangling::Utilities::cvImageToTensorConverter(rightCVImg, 128);

        std::vector<torch::Tensor> inferTensorVec;
        inferTensorVec.reserve(2);
        inferTensorVec.push_back(leftImageTensor);
        inferTensorVec.push_back(rightImageTensor);
        torch::Tensor stackedTensor = torch::stack(inferTensorVec);

        stackedTensor /= 255; // Normalization

        // Inferring
        torch::Tensor output = fln->forward(stackedTensor.to(device)).detach();

        // Convert output tensor to std::vector<std::tuple<float, float>>
        auto leftLabels = dataWrangling::Utilities::TensorToFloatVector(output[0], 500);
        auto rightLabels = dataWrangling::Utilities::TensorToFloatVector(output[1], 500);

        auto leftImageCVMat = dataWrangling::Utilities::TensorToCVMatConverter(leftImageTensor, 500);
        auto rightImageCVMat = dataWrangling::Utilities::TensorToCVMatConverter(rightImageTensor, 500);

        for (int i=0; i<leftLabels.size(); ++i){
            auto [leftX, leftY] = leftLabels[i];
            auto [rightX, rightY] = rightLabels[i];

            //if (Y < leftImageCVMat.rows and X < leftImageCVMat.cols) {
            //    cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 3, cv::Scalar( 54, 54, 251 ), cv::FILLED, cv::LINE_4);
            //}

            std::fprintf(stdout, "left: (%f, %f),\t right: (%f, %f)\n", leftX, leftY, rightX, rightY);
        }

        std::cout << std::endl;

        return std::make_tuple(leftImageCVMat, rightImageCVMat);
    }
}