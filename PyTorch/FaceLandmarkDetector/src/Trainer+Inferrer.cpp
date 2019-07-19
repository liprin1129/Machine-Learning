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
            fout << "/n";
        }

    }

    fout.clear();
}

void TrainerInferrer::train
(
    FaceLandmarkNet fln,
    bool verbose, torch::Device device, std::string imgFolderPath, std::string labelCsvFile, 
    float learningRate, int numEpoch, int numMiniBatch, int numWorkers,
    float wranglingProb, int resizeFactor, float contrastFactor, float brightnessFactor, float moveFactor,
    int saveInterval
)
{    
    //FaceLandmarkNet fln(3, verbose); // Num of channels, Verbose
    
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
                //_output = fln->forward(batch.data.to(device));

                torch::Tensor miniBatchLoss = torch::mse_loss(output, batch.target.to(device), Reduction::Mean);
                miniBatchLoss.backward(); // Calculate partial derivatives
                adamOptimizer.step(); // Back-propagation

                totLoss += miniBatchLoss.item<float>();
                //std::fprintf(stdout, "Batch Loss: %f, Total Loss: %f\n", miniBatchLoss.item<float>(), totLoss);
                ++batchCount;
                if ((epoch+1)%saveInterval == 0 and batchCount == 1) {
                    //std::cout << "SAVE!!!!\n";
                    //testShow(epoch+1, batch.data*255, output*batch.data.size(2));
                    testShow(epoch+1, 500, batch.data*255, output);
                }
            }
        }

        // Test
        std::fprintf(stdout, "Epoch #[%d/%d] | (Train total loss: %f)\n", epoch+1, numEpoch, (totLoss*sqrt(pow(resizeFactor, 2) + pow(resizeFactor, 2)))/batchCount);
        //batchCount = 0;
        
        if ((epoch+1) % saveInterval == 0) {
            //std::cout << "SAVE!\n";
            char saveModelString[100];
            char saveOptimString[100];

            std::sprintf(saveModelString, "./checkpoints/Trained-models/model-%03d.pt", epoch+1);
            std::sprintf(saveOptimString, "./checkpoints/Trained-models/optim-%03d.pt", epoch+1);
            
            torch::save(fln, saveModelString);
            torch::save(adamOptimizer, saveOptimString);
        }

        //if ((epoch+1) % (saveInterval*2) == 0) {
            //std::cout << "LOAD!\n";
        //    infer(epoch+1-saveInterval, numMiniBatch, resizeFactor, device);
        //}
    }
}


void TrainerInferrer::inferStereo(FaceLandmarkNet fln, const at::Tensor &leftImageTensor, const at::Tensor &rightImageTensor, torch::Device device) {    
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
