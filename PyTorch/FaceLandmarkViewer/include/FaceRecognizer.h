#pragma once

#include <torch/torch.h>
#include <DataWranglingHelper.h>

class FaceRecognizer {
    public:
        static void landmarkMeanDifferences(at::Tensor const &inTensor);
};