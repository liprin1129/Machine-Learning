#include "FaceRecognizer.h"

std::vector<std::string> FaceRecognizer::csvFinder(const std::string &rootDir) {
    std::vector<std::string> files;

    for (const auto & entry : filesystem::recursive_directory_iterator(rootDir)){
        //std::cout << entry.path() << std::endl;
        std::string filePath = entry.path();

        std::string csvPath;
        if (filePath.find("csv") != std::string::npos) {
            files.push_back(filePath);
        }
    }

    return files;
}

std::vector<std::vector<float>> FaceRecognizer::csvReader(const std::string &filePath) {
    //std::vector<std::tuple<std::string, std::vector<float>>> dataset;

    // File pointer
    std::fstream fin;

    // Open an existing record 
    fin.open(filePath, std::fstream::in); 

    std::vector<std::vector<float>> meanAndVarVec;
    std::vector<float> row;
    std::string line, word, temp;

    // Skip the first row
    std::getline(fin, line);
    //std::cout << line << std::endl;
    //std::getline(fin, line);
    //std::cout << line << std::endl;

    while (std::getline(fin, line)) { 
        row.clear();

        std::stringstream s(line);
  
        // read every column data of a row and 
        // store it in a string variable, 'word' 
        int count = 0;
        while (std::getline(s, word, ',')) { 
            //std::cout << word << " ";
            // add all the column data 
            // of a row to a vector 
            //std::cout << ++count << std::endl;
            row.push_back(stof(word)); 
        } 
    
        meanAndVarVec.push_back(row);
    }

    //std::cout << meanAndVarVec.size() << "\n";
    
    /*for (auto &c1: coordVec) {
        std::cout << c1.size() << std::endl;
        for (auto &c2: c1) {
            std::cout << c2 << " ";
        }
        std::cout << std::endl;
    }*/
    
    fin.close();

    return meanAndVarVec;
}

void FaceRecognizer::faceRecognition(FaceLandmarkNet fln, const at::Tensor &imageTensor, const std::vector<std::vector<float>> &meanAndVarVec, torch::Device device) {
    int centreIdx = 33; // centre number of landmarks

    fln->eval();                        // evaluation mode
    fln->to(device);                    // Upload the model to the device {CPU, GPU}

    // inferring landmarks given a image
    std::vector<torch::Tensor> testTensorVec;
    testTensorVec.reserve(2);
    testTensorVec.push_back(imageTensor);
    testTensorVec.push_back(imageTensor);
    torch::Tensor stackedTensor = torch::stack(testTensorVec);

    stackedTensor /= 255; // Normalization

    torch::Tensor output = fln->forward(stackedTensor.to(device)).detach();
    output = (output*1e+4).round()*1e-4; // round output at 4 decimal points
    //std::cout << output[0] << std::endl;

    // Calculate distances
    //      1) Seperate x and y coordinates
    //std::cout << output.sizes() << std::endl;
    //output.to(torch::Device(torch::kCPU));
    //auto a = output.accessor<float, 2>();

    torch::TensorOptions tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    at::Tensor xCoordTensor = torch::zeros({(int)output.size(1)/2}, tensorOptions);
    at::Tensor yCoordTensor = torch::zeros({(int)output.size(1)/2}, tensorOptions);


    for (int i=0; i<output.size(1); ++i) {
        if (i%2 == 0) {
            xCoordTensor[(int)(i/2)] = output[0][i] - output[0][centreIdx*2];
        }
        else {
            yCoordTensor[(int)(i/2)] = output[0][i];
        }
    }

    //std::cout << output[0] << std::endl;
    //std::cout << xCoordTensor << std::endl;
    //std::cout << yCoordTensor << std::endl;

    //      2) Normalize x and y coordinates [-1, 1]
    //std::cout << xCoordTensor.sizes() << std::endl;
    //std::cout << xCoordTensor.max() << std::endl;
    //std::cout << xCoordTensor.min() << std::endl;
    
    auto xNormCoordTensor = torch::zeros_like(xCoordTensor);
    auto yNormCoordTensor = torch::zeros_like(yCoordTensor);

    for (int i=0; i<xCoordTensor.size(0); ++i) {
        xNormCoordTensor[i] = 2*(xCoordTensor[i] - xCoordTensor.min()) / (xCoordTensor.max() - xCoordTensor.min())-1;
        yNormCoordTensor[i] = 2*(yCoordTensor[i] - yCoordTensor.min()) / (yCoordTensor.max() - yCoordTensor.min())-1;
    }

    //std::cout << xNormCoordTensor.min().item<float>() << ", " << xNormCoordTensor.max().item<float>() << std::endl;
    //std::cout << yNormCoordTensor.min().item<float>() << ", " << yNormCoordTensor.max().item<float>() << std::endl;


    //      3) Calculate distance
    auto distanceTensor = torch::zeros_like(yCoordTensor);

    for (int i=0; i< distanceTensor.size(0); ++i) {
        distanceTensor[i] = 
            torch::sqrt(
                torch::pow(xNormCoordTensor[i] - xNormCoordTensor[centreIdx], 2)
                + torch::pow(yNormCoordTensor[i] - yNormCoordTensor[centreIdx], 2)
                );
    }

    //std::cout << distanceTensor << std::endl;

    //      4) Predict
    int employIdx = 0;
    int employNum[] = {117, 129, 153, 155, 156, 157, 164, 166, 171};

    for (const auto &meanAndVar: meanAndVarVec){
        int numCoords = xCoordTensor.size(0); // total number of coordinates
        int distanceIdx = 0; // index for distance tensor
        int matchCount = 0; // variable to save the distance is between the distributions

        for (int i=(int)(meanAndVar.size()*(2/3.0)); i<meanAndVar.size(); i+=2){
            auto mean = meanAndVar[i];
            auto var = meanAndVar[i+1];
            bool trueFalseIdentifier = false;

            if ( ((mean - var) <= distanceTensor[distanceIdx]).item<bool>() and (distanceTensor[distanceIdx] <= (mean + var)).item<bool>()  ){
                ++matchCount;
                trueFalseIdentifier = true;
            }
            
            //std::cout << i << ", " << distanceIdx << "\t";
            //std::fprintf(stdout, "[%f <= %f <= %f] %s\n", 
            //    mean-var, distanceTensor[distanceIdx].item<float>(), mean+var,
            //    trueFalseIdentifier? "true":"false");
            
            ++distanceIdx;
        }

        std::fprintf(stdout, "%d: %d", employNum[employIdx++], --matchCount);

        distanceIdx = 0;
        matchCount = 0;

        std::cout << std::endl;
    }
}