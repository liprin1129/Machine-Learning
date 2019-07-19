#pragma once

#include <iostream>
#include <experimental/filesystem>
//#include <vector>
//#include <algorithm>
#include <fstream>
#include <string>


namespace filesystem = std::experimental::filesystem;

class FaceRecognizer {
    private:
        std::vector<std::vector<float>> meanAndVar;
        
    public:
        std::vector<std::string> csvFinder(const std::string &rootDir);
        std::vector<std::vector<float>> csvReader(const std::string &filePath);
        void csvReader(const std::vector<std::vector<float>> &coordVec);
};
