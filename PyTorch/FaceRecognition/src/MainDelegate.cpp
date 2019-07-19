#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){

    FaceRecognizer fr;
    auto csvFiles = fr.csvFinder("/DATASETs/Face/Face-SJC/Temp-Detection-Check-Data/");

    for (auto &csvFile: csvFiles) {
        auto b = fr.csvReader(csvFile);
        break;
    }
}
