#include "Trainer+Inferrer.h"
TrainerInferrer::TrainerInferrer(int numEpoch, int numBatch, std::tuple<int, int> imgRescale, bool verbose) {
    _imgRescale = imgRescale;
    _verbose = verbose;
    _numEpoch = numEpoch;
    _numBatch = numBatch;
    
    fln = FaceLandmarkNet(3, verbose);
}

void TrainerInferrer::train(torch::Device device, std::string imgFolderPath, std::string labelCsvFile) {
    fln->to(device); // Upload the model to the device {CPU, GPU}

    // Optimizer
    torch::optim::Adam adamOptimizer(
        fln->parameters(),
        torch::optim::AdamOptions(5e-5).beta1(0.5));

    auto cds = CustomDataset(
        "/DATASETs/Face/Landmarks/300W-Dataset/300W/face_landmarks.csv",
        "/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/",
        //"/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv",
        //"/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/",
        false)
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::Resize(500)))
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::RandomColour(0.7, 3.0, 50.0)))
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(dataWrangling::RandomClop(0.7, 100.0)))
        .map(torch::data::transforms::Stack<>());

    // Generate a data loader.
    //auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(cds), 
        torch::data::DataLoaderOptions().batch_size(_numBatch).workers(4));

    for (int epoch = 0; epoch < _numEpoch; ++epoch) {
        //std::fprintf(stdout, "Start epoch #%d\n", epoch);

        float totLoss = 0;
        int batchCount = 0;
        for (auto& batch : *data_loader) {
            auto data = batch.data;
            auto labels = batch.target;
            
            // std::cout << "Data is cuda?: " << data.to(device).is_cuda() << std::endl;
            // std::cout << "Labels is cuda?: " << labels.to(device).is_cuda() << std::endl;

            if (not(torch::isnan(data).sum().item<int>() or torch::isnan(labels).sum().item<int>())){
                
                adamOptimizer.zero_grad();
                torch::Tensor output = fln->forward(data.to(device));

                torch::Tensor miniBatchLoss = torch::mse_loss(output, labels.to(device), Reduction::Mean);
                
                miniBatchLoss.backward();
                adamOptimizer.step();

                totLoss += miniBatchLoss.item<float>();

                if ((epoch+1)%2 == 0 and batchCount++ == 0) {
                    //checkTensorImgAndLandmarks(epoch+1, data[0]*255, output[0]*std::get<0>(_imgRescale), labels[0]*std::get<0>(_imgRescale));
                    //std::fprintf(stdout, "Epoch #[%d/%d] | (Train total loss: %f)\n", epoch+1, _numEpoch, totLoss*std::get<0>(_imgRescale));
                    //checkTensorImgAndLandmarks(epoch+1, data[0]*255, output[0], labels[0], 500);
                    std::fprintf(stdout, "Epoch #[%d/%d] | (Train total loss: %f)\n", epoch+1, _numEpoch, totLoss);
                }
            }
            else {
                std::fprintf(stderr, "NAN value detected!\n");
                exit(-1);
            }
        }
        
        /*
        if ((epoch+1)%200 == 0 and epoch <= (int)_numEpoch*0.8) {
            char outputString[100];
            char optimizerString[100];
            std::sprintf(outputString, "./checkpoints/Trained-models/output-epoch%04d.pt", epoch+1);
            std::sprintf(optimizerString, "./checkpoints/Trained-models/optimizer-epoch%04d.pt", epoch+1);
            torch::save(fln, outputString);
            torch::save(adamOptimizer, optimizerString);
        }
        else if ((epoch+1)%100 == 0 and epoch > (int)_numEpoch*0.8 and epoch <= (int)_numEpoch*0.95) {
            char outputString[100];
            char optimizerString[100];
            std::sprintf(outputString, "./checkpoints/Trained-models/output-epoch%04d.pt", epoch+1);
            std::sprintf(optimizerString, "./checkpoints/Trained-models/optimizer-epoch%04d.pt", epoch+1);
            torch::save(fln, outputString);
            torch::save(adamOptimizer, optimizerString);
        }
        else if ((epoch+1) > (int)_numEpoch*0.95) {
            char outputString[100];
            char optimizerString[100];
            std::sprintf(outputString, "./checkpoints/Trained-models/output-epoch%04d.pt", epoch+1);
            std::sprintf(optimizerString, "./checkpoints/Trained-models/optimizer-epoch%04d.pt", epoch+1);
            torch::save(fln, outputString);
            torch::save(adamOptimizer, optimizerString);
        }
        */
    }
}

void TrainerInferrer::infer(torch::Device device, std::string imgPath, std::string modelPath) {
    // Read image
    cv::Mat img = cv::imread(imgPath);
    img.convertTo(img, CV_32FC3); // Convert CV_8UC3 data type to CV_32FC3

    // Resize image
    auto [newW, newH] = _imgRescale;
    cv::Mat resizedImage;
    cv::resize(img, resizedImage, cv::Size2d(newH, newW), 0, 0, cv::INTER_LINEAR);
    img = resizedImage/255; // rescale to [0, 1]

    // Convert to Tensor
    torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, imgOptions);
    imgTensor = imgTensor.permute({2, 0, 1}); // convert to CxHxW
    imgTensor = imgTensor.unsqueeze(0); // If the image is 3 dimensions, add extra pseudo-dimension

    torch::load(fln, modelPath);
    at::Tensor output = fln->forward(imgTensor.to(device));

    checkTensorImgAndLandmarks(imgTensor[0], output, 500);
}

void TrainerInferrer::checkTensorImgAndLandmarks(torch::Tensor const &imgTensor, torch::Tensor const &inferredTensor, int newWH) {
    auto copiedImgTensor = (imgTensor*255).toType(torch::kUInt8).clone();
    
    std::cout << imgTensor.sizes() << std::endl;
    int cvMatSize[2] = {(int)imgTensor.size(1), (int)imgTensor.size(2)};
    cv::Mat imgCVB(2, cvMatSize, CV_8UC1, copiedImgTensor[0].data_ptr());
    cv::Mat imgCVG(2, cvMatSize, CV_8UC1, copiedImgTensor[1].data_ptr());
    cv::Mat imgCVR(2, cvMatSize, CV_8UC1, copiedImgTensor[2].data_ptr()); //CV_8UC1
    //std::cout << "Pass: Tensor to cv::Mat" << std::endl;

    // Merge each channel to create colour cv::Mat
    cv::Mat imgCV; // Merged output cv::Mat
    std::vector<cv::Mat> channels;
    channels.push_back(imgCVB);
    channels.push_back(imgCVG);
    channels.push_back(imgCVR);
    cv::merge(channels, imgCV);
    
    cv::resize(imgCV, imgCV, cv::Size2d(newWH, newWH), 0, 0, cv::INTER_LINEAR);
    //std::cout << "Pass: cv::Mat merged" << std::endl;

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;
    for (int i=0; i<inferredTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = inferredTensor[0][i].item<float>()*newWH;//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 2, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = inferredTensor[0][i].item<float>()*newWH;//*outputImg.cols;
    }

    cv::namedWindow("Restored", CV_WINDOW_AUTOSIZE);
    imshow("Restored", imgCV);
    cv::waitKey(0);
}

void TrainerInferrer::checkTensorImgAndLandmarks(int epoch, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor, torch::Tensor const &gtLabelTensor, int newWH) {
    //std::cout << imgTensor.sizes() << std::endl;
    //std::cout << labelTensor.sizes() << std::endl<<std::endl;
    
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
    //imgCV.convertTo(imgCV, CV_8UC3);

    //cv::Mat resizedImage;
    cv::resize(imgCV, imgCV, cv::Size2d(newWH, newWH), 0, 0, cv::INTER_LINEAR);

    // Convert the ground truth label Tensor to vector
    std::vector<std::tuple<float, float>> gtLandmarks;
    float X = 0.0, Y=0.0;
    for (int i=0; i<gtLabelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = gtLabelTensor[0][i].item<float>()*newWH;//*outputImg.rows;
            gtLandmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 2, cv::Scalar( 0, 255, 0 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = gtLabelTensor[0][i].item<float>()*newWH;//*outputImg.cols;
    }

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    X = 0.0, Y=0.0;
    for (int i=0; i<labelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = labelTensor[0][i].item<float>()*newWH;//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 2, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = labelTensor[0][i].item<float>()*newWH;//*outputImg.cols;
    }

    char outputString[100];
    std::sprintf(outputString, "./checkpoints/Images/output-epoch%03d.jpg", epoch);
    cv::imwrite( outputString, imgCV );
    
    /*
    cv::namedWindow("Restored", CV_WINDOW_AUTOSIZE);
    imshow("Restored", imgCV);
    cv::waitKey(0);
    */
}
