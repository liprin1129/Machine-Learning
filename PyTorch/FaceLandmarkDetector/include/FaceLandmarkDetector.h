#ifndef __FACE_LANDMARK_DETECTOR_H__
#define __FACE_LANDMARK_DETECTOR_H__

#include <torch/torch.h>
#include <iostream>

class FaceLandmarkDetector : public torch::nn::Module {
    //private:
    public:
        int64_t inputChannel;

        torch::nn::Conv2d conv1{nullptr};//, conv2, conv3, conv4;

    //public:
        FaceLandmarkDetector();
};

#endif //__FACE_LANDMARK_DETECTOR_H__