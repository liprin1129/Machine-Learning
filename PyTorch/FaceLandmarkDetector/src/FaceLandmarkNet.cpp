#include "FaceLandmarkNet.h"

// ************************ //
// * FaceLandmarNet class * //
// ************************ //

FaceLandmarkNetImpl::FaceLandmarkNetImpl(int numEpoch, int numBatch, std::tuple<int, int> imgRescale, bool verbose, bool testFlag) {
    std::cout << "Constructor" << std::endl;
    
    _verbose = verbose;
    _testFlag = testFlag;

    inputChannel = 3;
    _numEpoch = numEpoch;
    _numBatch = numBatch;
    _imgRescale = imgRescale;

    // Convolutional layer #1: 
    // [channel: input -> 32], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannel, 32, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm1 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(32));
    
    // Convolutional layer #2: 
    // [channel: 32 -> 64], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm2 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(64));

    // Convolutional layer #3: 
    // [channel: 64 -> 64], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm3 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(64));

    // Convolutional layer #4: 
    // [channel: 64 -> 64], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm4 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(64));

    // Convolutional layer #5: 
    // [channel: 64 -> 64], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm5 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(64));

    // Convolutional layer #6: 
    // [channel: 64 -> 128], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv6 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm6 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(128));

    // Convolutional layer #7: 
    // [channel: 128 -> 256], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv7 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm7 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(128));

    // Convolutional layer #7: 
    // [channel: 128 -> 128], [filter: 3x3], [stride: 1x1], [padding: 0]
    conv8 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3)
                .stride(1)
                .padding(0)
                .with_bias(false));
    batch_norm8 = torch::nn::BatchNorm(torch::nn::BatchNormOptions(256));

    // [1 x 1] Convolutional layer #8:
    // [channel: 256 -> 68], [filter: 1x1], [stride: 1x1], [padding:0]
    convOneXOne1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 136, 1)
                .stride(1)
                .padding(0)
                .with_bias(true));

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("conv5", conv5);
    register_module("conv6", conv6);
    register_module("conv7", conv7);
    register_module("conv8", conv8);
    register_module("convOneXOne1", convOneXOne1);
    
    
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
    register_module("batch_norm4", batch_norm4);
    register_module("batch_norm5", batch_norm5);
    register_module("batch_norm6", batch_norm6);
    register_module("batch_norm7", batch_norm7);
    register_module("batch_norm8", batch_norm8);
    
}

torch::Tensor FaceLandmarkNetImpl::forward(torch::Tensor x, bool trainFlag) {

    // Layer #1
    if (_verbose) std::cout << "Layer #1:\n";
    if (_verbose) std::cout << "\t Input: \t" << x.sizes() << std::endl;

    if (trainFlag) x = torch::relu(batch_norm1(conv1(x)));
    else x = torch::relu(conv1(x));
    if (_verbose) std::cout << "\t Conv1: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;
    
    // Layer #2
    if (_verbose) std::cout << "Layer #2:\n";
    
    if (trainFlag) x = torch::relu(batch_norm2(conv2(x)));
    else x = torch::relu(conv2(x));
    if (_verbose) std::cout << "\t Conv2: \t" << x.sizes() << std::endl;
    
    if (trainFlag) x = torch::relu(batch_norm3(conv3(x)));
    else x = torch::relu(conv3(x));
    if (_verbose) std::cout << "\t Conv3: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // Layer #3
    if (_verbose) std::cout << "Layer #3:\n";
    
    if (trainFlag) x = torch::relu(batch_norm4(conv4(x)));
    else x = torch::relu(conv4(x));
    if (_verbose) std::cout << "\t Conv4: \t" << x.sizes() << std::endl;
    
    if (trainFlag) x = torch::relu(batch_norm5(conv5(x)));
    else x = torch::relu(conv5(x));
    if (_verbose) std::cout << "\t Conv5: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // Layer #4
    if (_verbose) std::cout << "Layer #4:\n";
    
    if (trainFlag) x = torch::relu(batch_norm6(conv6(x)));
    else x = torch::relu(conv6(x));
    if (_verbose) std::cout << "\t Conv6: \t" << x.sizes() << std::endl;
    
    if (trainFlag) x = torch::relu(batch_norm7(conv7(x)));
    else x = torch::relu(conv7(x));
    if (_verbose) std::cout << "\t Conv7: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // Layer #5
    if (_verbose) std::cout << "Layer #5:\n";
    
    //if (trainFlag) x = torch::relu(batch_norm8(conv8(x)));
    //else x = torch::relu(conv8(x));
    x = torch::relu(conv8(x));
    if (_verbose) std::cout << "\t Conv8: \t" << x.sizes() << std::endl;
    
    x = torch::adaptive_avg_pool2d(x, {1, 1});
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // Layer #6
    if (_verbose) std::cout << "Layer #6:\n";

    //x = torch::relu(convOneXOne1(x));
    //x = torch::leaky_relu(convOneXOne1(x));
    x = convOneXOne1(x);
    if (_verbose) std::cout << "\t 1x1 Conv: \t" << x.sizes() << std::endl;

    // Squeeze
    x = x.squeeze();
    //x = torch::softmax(x.unsqueeze(1), 2);
    
    if (_numBatch > 1) { // in the case that batch_size is greater than 1
        x = torch::tanh(x.unsqueeze(1));
    }

    else { // if batch_size is 1.
        x = torch::tanh(x.unsqueeze(0));
        x = x.unsqueeze(0);
    }

    if (_verbose) std::cout << "Last: \n";
    if (_verbose) std::cout << "\t output: \t" << x.sizes() << std::endl;

    return x;
}

void FaceLandmarkNetImpl::train(torch::Device device, torch::optim::Optimizer &optimizer) {
    this->to(device);

    auto cds = CustomDataset(
        "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/face_landmarks.csv", 
        "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/",
        _imgRescale, 
        _verbose)
        //.map(torch::data::transforms::Normalize<>(-0.5, 1))
        .map(torch::data::transforms::Stack<>());

    // Generate a data loader.
    //auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(cds), 
        torch::data::DataLoaderOptions().batch_size(_numBatch).workers(1));

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
                
                optimizer.zero_grad();
                torch::Tensor output = forward(data.to(device), true);
                //std::fprintf(stdout, "(output) Max: %f, Min: %f\n", output.max().item<float>(), output.min().item<float>());
                
                torch::Tensor miniBatchLoss = torch::mse_loss(output, labels.to(device), Reduction::Mean);
                //std::fprintf(stdout, "(output) Loss: %f\n", miniBatchLoss.item<float>());

                miniBatchLoss.backward();
                optimizer.step();

                totLoss += miniBatchLoss.item<float>();
                //std::fprintf(stdout, "\t(loss: %f)\n", totLoss);

                if (epoch%10 == 0 and batchCount++ == 0) {
                    checkTensorImgAndLandmarks(epoch, data[0]*255, output[0]*std::get<0>(_imgRescale), labels[0]*std::get<0>(_imgRescale));
                    std::fprintf(stdout, "Epoch #[%d/%d] | (Train loss: %f)\n", epoch+1, _numEpoch, totLoss);
                    std::fprintf(stdout, "\t[Labels] Mean: %f, Var: %f\n", labels[0].mean().item<float>(), labels[0].std().item<float>());
                    std::fprintf(stdout, "\t[output] Mean: %f, Var: %f\n\n", output[0].mean().item<float>(), output[0].std().item<float>());
                }
                
                break;
            }
            else {
                std::fprintf(stderr, "NAN value detected!\n");
                exit(-1);
            }
        }
        break;
    }
}


void FaceLandmarkNetImpl::checkTensorImgAndLandmarks(int epoch, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor, torch::Tensor const &gtLabelTensor) {
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

    // Convert the ground truth label Tensor to vector
    std::vector<std::tuple<float, float>> gtLandmarks;
    float X = 0.0, Y=0.0;
    for (int i=0; i<gtLabelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = gtLabelTensor[0][i].item<float>();//*outputImg.rows;
            gtLandmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 3, cv::Scalar( 0, 255, 0 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = gtLabelTensor[0][i].item<float>();//*outputImg.cols;
    }

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    X = 0.0, Y=0.0;
    for (int i=0; i<labelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = labelTensor[0][i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 2, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = labelTensor[0][i].item<float>();//*outputImg.cols;
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