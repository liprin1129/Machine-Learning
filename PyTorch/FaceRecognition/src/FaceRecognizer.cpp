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

    std::vector<std::vector<float>> coordVec;
    std::vector<float> row;
    std::string line, word, temp;

    while (fin >> temp) { 
  
        row.clear(); 
  
        // read an entire row and 
        // store it in a string variable 'line' 
        std::getline(fin, line); 
  
        // used for breaking words 
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
    
        coordVec.push_back(row);
    }

    //std::cout << coordVec.size() << "\n";
    
    /*for (auto &c1: coordVec) {
        std::cout << c1.size() << std::endl;
        for (auto &c2: c1) {
            std::cout << c2 << " ";
        }
        std::cout << std::endl;
    }*/
    
    fin.close();

    return coordVec;
}

void FaceRecognizer(const std::vector<std::vector<float>> &coordVec) {
    
}