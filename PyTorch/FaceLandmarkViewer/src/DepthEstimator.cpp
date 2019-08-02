#include "DepthEstimator.h"

DepthEstimator::DepthEstimator() {
    DepthEstimatorHasLoaded();
}

// Compute Z coordinate (This is not a good algorithm. This is only for simple verification.)
void DepthEstimator::estimateCoordinates3D(
    const std::vector<std::tuple<float, float>> &leftCoordinates2D,
    const std::vector<std::tuple<float, float>> &rightCoordinates2D, 
    const int &lFocalLengthX,
    const int &lFocalLengthY,
    const int &lOpticalCentreX,
    const int &lOpticalCentreY,
    const int &focalLength) {
    
    //std::vector<std::list<int>> 
    //std::vector<float[3]> coordinates3DVec;
    //std::fprintf(stdout, "[%f, %f]\n", std::get<0>(leftCoordinates2D[33]), std::get<0>(rightCoordinates2D[33]));

    for (int idx=0; idx<leftCoordinates2D.size(); ++idx) {
        auto [lX, lY] = leftCoordinates2D[idx];
        auto [rX, rY] = rightCoordinates2D[idx];

        //float coordinates3DArr[3] = {0, 0, 0};
        _coordinate3DArr[idx][2] = static_cast<float>(120)*focalLength / abs(rX-lX); // compute Z coordinate, 120 is baseline in cm
        _coordinate3DArr[idx][0] = (lX - lOpticalCentreX)*_coordinate3DArr[idx][2] / lFocalLengthX; // X coordinate
        _coordinate3DArr[idx][1] = (lY - lOpticalCentreY)*_coordinate3DArr[idx][2] / lFocalLengthY; // Y coordinate
    }
}

void DepthEstimator::incrementalMean(int &clockCount) {
    auto coordTensor = DataWrangling::Utilities::coordArrToTensorConverter(_coordinate3DArr);
    //std::cout << "coordTensor\n" << coordTensor << std::endl;

    // Normalize coordTensor into norm Tensor within [0, 1]
    auto normTensor = DataWrangling::Utilities::normMinMax(coordTensor);
    //std::cout << "Norm\n" << normTensor << std::endl;

    if (_classCount <= 5.0) {
        _initMeanVarTensor(normTensor);
        _classCount += 1.0;
    }
    else {
        //std::cout << (torch::sum(torch::pow(_oldCoordTensor-normTensor, 2)).item<float>()) << std::endl;
        auto l2Distance = torch::sum(torch::pow(_oldCoordTensor-normTensor, 2)).item<float>();
        
        if ( l2Distance < 5.0 ) {
            auto newMeanTensor = _meanTensor + (normTensor - _meanTensor) / _classCount;
            //auto newVarTensor = torch::sqrt( ( torch::pow(_varTensor, 2)*_classCount + (normTensor-_meanTensor)*(normTensor-newMeanTensor) ) / _classCount);
            
            // Update
            _meanTensor = newMeanTensor;
            //std::cout << "[O] UPDATE" << std::endl;

            _classCount += 1.0;
            clockCount--;
        }
        else {
            //std::cout << "[X] NO UPDATE >>" << std::endl;
        }
    }

    // Update
    _oldCoordTensor = normTensor;

    /*auto idx = 33;
    auto X = _coordinate3DArr[idx][0];
    auto Y = _coordinate3DArr[idx][1];
    auto Z = _coordinate3DArr[idx][2];

    if (_classCount <= 20) {
        _oldX = X;
        _oldY = Y;
        _oldZ = Z;

        _updateFlag = true;
    }

    if (abs(_oldX-X)<100 and abs(_oldY-Y)<100 and abs(_oldZ-Z)<100) { // unit is cm
        // Compute means
        auto newMeanX = _meanX + static_cast<float>(X-_meanX) / _classCount;
        auto newMeanY = _meanY + static_cast<float>(Y-_meanY) / _classCount;
        auto newMeanZ = _meanZ + static_cast<float>(Z-_meanZ) / _classCount;

        // Compute variances
        auto newVarX = sqrt(static_cast<float>(pow(_varX, 2.0)*_classCount + (X-_meanX)*(X-newMeanX)) / _classCount);
        auto newVarY = sqrt(static_cast<float>(pow(_varY, 2.0)*_classCount + (Y-_meanY)*(Y-newMeanY)) / _classCount);
        auto newVarZ = sqrt(static_cast<float>(pow(_varZ, 2.0)*_classCount + (Z-_meanZ)*(Z-newMeanZ)) / _classCount);

        // Update
        _meanX = newMeanX;
        _meanY = newMeanY;
        _meanZ = newMeanZ;

        _varX = newVarX;
        _varY = newVarY;
        _varZ = newVarZ;

        _oldX = X;
        _oldY = Y;
        _oldZ = Z;
        
        ++_classCount;

        _updateFlag = true;
    }
    else {
        _updateFlag = false;
    }

    std::fprintf(stdout, "%s [X, Y, Z]:[%f, %f, %f], \t Means:[%f, %f, %f], \t Variances[%f, %f, %f]\n",
        _updateFlag==true? "True":"False", X, Y, Z, _meanX, _meanY, _meanZ, _varX, _varY, _varZ);
    */
}


void DepthEstimator::_initMeanVarTensor(at::Tensor const &inVecTensor) {
        _meanTensor = inVecTensor.mean();
        //_covTensor = torch::zeros(inVecTensor.size(0));
}


void DepthEstimator::_initMeanCovTensor(at::Tensor const &inVecTensor) {
        _meanTensor = inVecTensor.mean();
        _covTensor = torch::eye(inVecTensor.size(0));
}


void DepthEstimator::_absDistanceCalculator() {
    auto secondOrder = torch::pow(_meanTensor, 2);
    _distanceTensor = torch::sqrt(secondOrder.sum(1));
    //std::cout << _distanceTensor.sizes() << std::endl;
}


int DepthEstimator::DepthEstimatorHasLoaded() {
    /*
    _meanX = 0;
    _meanY = 0;
    _meanZ = 0;

    _varX = 0;
    _varY = 0;
    _varZ = 0;

    _oldX = 0;
    _oldY = 0;
    _oldZ = 0;
    
    _classCount=1;

    _updateFlag = true;
    */
   _classCount=1.0;
}