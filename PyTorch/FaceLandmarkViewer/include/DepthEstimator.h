#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <tuple>
#include <cmath>

#include <torch/torch.h>

class DepthEstimator {
    private:
        //std::vector<std::list<int>> _Coordinates3D;
        float _coordinate3DArr[68][3]; // array of 68 landmarks and [X, Y, Z] coordinates

        torch::Tensor _oldTensor;

        /*
        int _oldX, _oldY, _oldZ;
        float _meanX, _meanY, _meanZ;
        float _varX, _varY, _varZ;
        int64_t _classCount; // total count number to calculate mean and variance
        bool _updateFlag;
        */

    public:
        // Constructor
        DepthEstimator();

        // Getters
        //int64_t getClassCount() {return _classCount;};
        //bool isUpdateFlagTrue() {return _updateFlag;};
        //std::tuple<float, float> getOldXandY() {return std::make_tuple(_oldX, _oldY);};

        //std::vector<std::list<int>>
        void estimateCoordinates3D(
            const std::vector<std::tuple<float, float>> &leftCoordinates2D, 
            const std::vector<std::tuple<float, float>> &rightCoordinates2D,
            const int &lFocalLengthX,
            const int &lFocalLengthY,
            const int &lOpticalCentreX,
            const int &lOpticalCentreY,
            const int &focalLength
        );

        void incrementalMeanAndVariance();
        int DepthEstimatorHasLoaded();
};