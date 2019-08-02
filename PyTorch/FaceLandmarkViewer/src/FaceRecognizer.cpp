#include "FaceRecognizer.h"

void FaceRecognizer::landmarkMeanDifferences(at::Tensor const &inTensor) {
    
    std::vector<at::Tensor> tempVec;
    tempVec.push_back(DataWrangling::Utilities::readCSVtoTensor("129"));
    tempVec.push_back(DataWrangling::Utilities::readCSVtoTensor("164"));
    tempVec.push_back(DataWrangling::Utilities::readCSVtoTensor("170"));
    tempVec.push_back(DataWrangling::Utilities::readCSVtoTensor("171"));
    
    at::Tensor meanSumTensor = torch::zeros({(int)tempVec.size(), 3});
    for (int i=0; i<(int)tempVec.size(); ++i) {
        auto meanDiff = inTensor - tempVec[i];
        auto l2MeanDiff = torch::sqrt(torch::pow(meanDiff, 2));        
        //meanSumTensor[i] = l2MeanDiff.sum();
        for (int j=0; j<3; ++j) {
            meanSumTensor[i][j] = DataWrangling::Utilities::tensorColumnSlicer(l2MeanDiff, j).sum().item<float>();
        }
    }

    std::fprintf(stdout, "\n\nCoord Means:\n%.10f, %.10f, %.10f\n%.10f, %.10f, %.10f\n%.10f, %.10f, %.10f \n%.10f, %.10f, %.10f\n", 
        meanSumTensor[0][0].item<float>(), meanSumTensor[0][1].item<float>(), meanSumTensor[0][2].item<float>(),
        meanSumTensor[1][0].item<float>(), meanSumTensor[1][1].item<float>(), meanSumTensor[1][2].item<float>(),
        meanSumTensor[2][0].item<float>(), meanSumTensor[2][1].item<float>(), meanSumTensor[2][2].item<float>(),
        meanSumTensor[3][0].item<float>(), meanSumTensor[3][1].item<float>(), meanSumTensor[3][2].item<float>());

    at::Tensor recognizerTensor = torch::zeros((int)tempVec.size());
    float a = 0.3;
    float b = 0.3;
    float c = 0.3;

    for (int i=0; i<(int)tempVec.size(); ++i) {
        recognizerTensor[i] = a*meanSumTensor[i][0] + b*meanSumTensor[i][1] + c*meanSumTensor[i][2];
    }
    std::fprintf(stdout, "Processed Value:\n%f, %f, %f, %f\n", 
        recognizerTensor[0].item<float>(), 
        recognizerTensor[1].item<float>(), 
        recognizerTensor[2].item<float>(), 
        recognizerTensor[3].item<float>());

    std::vector<std::string> employs;
    employs.push_back("129");
    employs.push_back("164");
    employs.push_back("170");
    employs.push_back("171");
    
    std::fprintf(stdout, "\nResult:\n >>> %s\n\n", employs[recognizerTensor.argmin().item<int>()].c_str());
}