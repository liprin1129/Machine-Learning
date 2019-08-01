#pragma once

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

namespace DataWrangling {

class Utilities {
    public:
        // Normalize x and y coordinates [min, max]
        static at::Tensor normMinMax(at::Tensor const &inTensor) {
            //torch::TensorOptions normTensorOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false).device(torch::kCPU);
            //auto normTensor = torch::zeros_like(inTensor, normTensorOptions);

            //std::cout << inTensor.min_values(0) << std::endl;
            auto minCoordTensor = inTensor.min_values(0);
            auto maxCoordTensor = inTensor.max_values(0);

            auto normTensor = (inTensor - minCoordTensor) / (maxCoordTensor - minCoordTensor);
            //std::cout << "FIRST\n" << normTensor << std::endl;

            return normTensor;
        }

        static at::Tensor coordArrToTensorConverter(float inArr[][3]) {
            // Convert _coordinate3DArr[68][3] to torch::Tensor
            torch::TensorOptions outputTensorOptions = torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false).device(torch::kCPU);
            torch::Tensor outputTensor = torch::randn({68, 3}, outputTensorOptions);

            for (int i=0; i<68; ++i) {
                for (int j=0; j<3; ++j) {
                    outputTensor[i][j] = inArr[i][j];
                }
            }

            return outputTensor;
        }

        //static void writeMeanAndVarTensorToCSV(at::Tensor const &inMeanTensor, at::Tensor const &inVarTensor, std::string csvName) {
        static void writeMeanAndVarTensorToCSV(at::Tensor const &inMeanTensor, std::string csvName) {
            //std::cout << inMeanTensor << std::endl << inVarTensor << std::endl;
            
            std::ofstream fileStream;
            fileStream.open(csvName+".csv");
            //fileStream << "Landmark No., Mean: X, Variance: X, Mean: Y, Variance: Y, Mean: Z, Variance: Z\n";
            fileStream << "Landmark No., Mean: X, Mean: Y, Mean: Z\n";
            
            for (int i=0; i<inMeanTensor.size(0); ++i) {
                fileStream << i+1;

                for (int j=0; j<inMeanTensor.size(1); ++j) {
                    //std::cout << inMeanTensor[i][j].item<float>();
                    fileStream << ", " << inMeanTensor[i][j].item<float>();
                    //fileStream << ", " << inVarTensor[i][j].item<float>();
                }
                fileStream << "\n";
            }
            
            fileStream.close();
        }

        static at::Tensor tensorColumnSlicer(at::Tensor const &inTensor, int columnIdx) {
            return inTensor.index(
                {torch::arange(0, inTensor.size(0), torch::kLong), 
                torch::ones(inTensor.size(0), torch::kLong)*columnIdx});
        }

        //static void multivariateNormalDistribution(at::Tensor const &inTensor, at::Tensor const &inMeanTensor, at::Tensor const &inCovTensor) {
            
        //}
};

} // END namespace DataWrangling