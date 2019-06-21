#include "FaceLandmarkNet.h"

// ************************ //
// * FaceLandmarNet class * //
// ************************ //

FaceLandmarkNetImpl::FaceLandmarkNetImpl(bool verbose) {
    std::cout << "Constructor" << std::endl;
    
    _verbose = verbose;

    inputChannel = 3;

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

torch::Tensor FaceLandmarkNetImpl::forward(torch::Tensor x) {
    // Layer #1
    if (_verbose) std::cout << "Layer #1:\n";
    if (_verbose) std::cout << "\t Input: \t" << x.sizes() << std::endl;

    x = torch::relu(batch_norm1(conv1(x)));
    //x = torch::relu(conv1(x));
    if (_verbose) std::cout << "\t Conv1: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;
    
    // Layer #2
    if (_verbose) std::cout << "Layer #2:\n";
    
    x = torch::relu(batch_norm2(conv2(x)));
    //x = torch::relu(conv2(x));
    if (_verbose) std::cout << "\t Conv2: \t" << x.sizes() << std::endl;
    
    x = torch::relu(batch_norm3(conv3(x)));
    //x = torch::relu(conv3(x));
    if (_verbose) std::cout << "\t Conv3: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // Layer #3
    if (_verbose) std::cout << "Layer #3:\n";
    
    x = torch::relu(batch_norm4(conv4(x)));
    //x = torch::relu(conv4(x));
    if (_verbose) std::cout << "\t Conv4: \t" << x.sizes() << std::endl;
    
    x = torch::relu(batch_norm5(conv5(x)));
    //x = torch::relu(conv5(x));
    if (_verbose) std::cout << "\t Conv5: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // Layer #4
    if (_verbose) std::cout << "Layer #4:\n";
    
    x = torch::relu(batch_norm6(conv6(x)));
    //x = torch::relu(conv6(x));
    if (_verbose) std::cout << "\t Conv6: \t" << x.sizes() << std::endl;
    
    x = torch::relu(batch_norm7(conv7(x)));
    //x = torch::relu(conv7(x));
    if (_verbose) std::cout << "\t Conv7: \t" << x.sizes() << std::endl;
    
    x = torch::max_pool2d(x, {2, 2}, {2, 2}, 0);
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // Layer #5
    if (_verbose) std::cout << "Layer #5:\n";
    
    x = torch::relu(batch_norm8(conv8(x)));
    //x = torch::relu(conv8(x));
    if (_verbose) std::cout << "\t Conv8: \t" << x.sizes() << std::endl;
    
    x = torch::adaptive_avg_pool2d(x, {1, 1});
    if (_verbose) std::cout << "\t Max pool: \t" << x.sizes() << std::endl;

    // Layer #6
    if (_verbose) std::cout << "Layer #6:\n";
    x = convOneXOne1(x);
    if (_verbose) std::cout << "\t 1x1 Conv: \t" << x.sizes() << std::endl;

    // Squeeze
    x = x.squeeze();
    //x = torch::softmax(x.unsqueeze(0), 1);
    x = x.unsqueeze(0);
    if (_verbose) std::cout << "Last: \n";
    if (_verbose) std::cout << "\t output: \t" << x.sizes() << std::endl;

    return x;
}

void FaceLandmarkNetImpl::train(DataLoader &dl, torch::Device device, torch::optim::Optimizer &optimizer) {
    this->to(device);
   
    for (int epoch = 0; epoch < 100; ++epoch) {
        //torch::Tensor miniBatchLoss = torch::zeros(1, device);
        int count = 0;

        for (auto &data: dl.getDataset()) { // Image and Label iterator
            auto [cvImg, listLabel] = dl.loadOneTraninImageAndLabel(data, true);
            
            at::Tensor inX = cvMat2Tensor(cvImg, device); // Convert cv::Mat to Tensor
            
            // Convert labels to Tensor
            at::Tensor label = floatList2Tensor(listLabel, device);
            /*float labelsArr[listLabel.size()];
            std::copy(listLabel.begin(), listLabel.end(), labelsArr);
            torch::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
            torch::Tensor label = torch::from_blob(labelsArr, {1, (signed long) listLabel.size()}, labelOptions).to(device);*/

            std::cout << "NAN!!!!!: " << torch::isnan(inX).sum().item<int>() << " | " << torch::isnan(inX).sum().item<int>() << std::endl;

            if (_verbose) showTrainInfo(cvImg, listLabel, inX, label);
            
            if ((cvImg.cols < 1100) and (cvImg).rows < 1100) {
                optimizer.zero_grad();
                torch::Tensor output = forward(inX);
                //std::fprintf(stdout, "(Epoch #%d, Count #%d) | (sum: %f)\n", epoch, count, output.sum().item<float>());
                //std::cout << output << std::endl;

                torch::Tensor loss = torch::mse_loss(output, label, Reduction::Mean);
                loss.backward();
                optimizer.step();

                std::fprintf(stdout, "(Epoch #%d, Count #%d) | (sum: %f, loss: %f)\n", epoch, count, output.sum().item<float>(), loss.item<float>());
                //std::cout << loss << std::endl;
                ++count;
                /*if (count % 100) {
                //std::fprintf(stdout, "Epoch #%d: Mini Batch #%d (loss: %f)\n", epoch, miniBatchCounter, loss.item<float>());
                std::fprintf(stdout, "Epoch #%d, totCount #%d | loss: %f\n", epoch, count, loss.item<float>());
                //std::fprintf(stdout, "totCount #%d\n", totCount);
                }*/
            }
        }
        //break;
    }
}

at::Tensor FaceLandmarkNetImpl::cvMat2Tensor(cv::Mat cvMat, torch::Device device) {
    // Convert cv::Mat to Tensor
    torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    return torch::from_blob(
                    cvMat.data, 
                    {1, 3, cvMat.cols, cvMat.rows}, 
                    imgOptions).to(device);
}

at::Tensor FaceLandmarkNetImpl::floatList2Tensor(std::list<float> floatLists, torch::Device device) {
    float labelsArr[floatLists.size()];
    std::copy(floatLists.begin(), floatLists.end(), labelsArr);
    torch::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor labelTensor = torch::from_blob(labelsArr, {1, (signed long) floatLists.size()}, labelOptions).to(device);
    
    return labelTensor;
}

void FaceLandmarkNetImpl::showTrainInfo(cv::Mat cvImg, std::list<float> listLabel, at::Tensor &inX, at::Tensor &label) {
    // Check memory locations
    std::fprintf(stdout, "Memory location: \n\t%p | %p\n", &cvImg, &listLabel);
    // Check the loaded image data is OK!
    double min, max;
    cv::minMaxIdx(cvImg, &min, &max);
    std::fprintf(stdout, "Image Info: \n\tMin: %f | Max: %f \n", min, max);
    //for (auto &l: litLabel) std::fprintf(stdout, "%lf, ", l);

    // Check the loaded labels are OK!
    float minLable=1.0, maxLabel=0.0;
    for (auto &l: listLabel) {
    if (l < minLable) {
        minLable = l;
    }

    if (l > maxLabel) {
        maxLabel = l;
    }
    }
    std::fprintf(stdout, "Labels Info: \n\tMin: %f | Max: %f\n", minLable, maxLabel);

    // Converted image to tensor information
    std::cout << "inX Tensor Info.:" << std::endl;
    std::cout << "\t size: " << inX.sizes() << std::endl;
    std::cout << "\t max: " << inX.max() << std::endl;
    std::cout << "\t min: " << inX.min() << std::endl;
    

    // Converted lables to tensor information
    std::cout << "labels Tensor Info.:" << std::endl;
    std::cout << "\t size: " << label.sizes() << std::endl;
    std::cout << "\t max: " << label.max() << std::endl;
    std::cout << "\t min: " << label.min() << std::endl;
    std::cout << std::endl;
}
