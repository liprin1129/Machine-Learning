#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
    /*
    auto inX = torch::randn({1, 3, 256, 128});
    //std::cout << inX.sizes() << std::endl;

    FaceLandmarkDetector fld;
    std::cout << fld.getConv() << std::endl;
    fld.forward(inX);
    */

   ImageDataLoader idl;
   idl.readLabel();
}