//#ifndef JSONFILEMANAGER_H
//#define JSONFILEMANAGER_H

#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <vector>

class JsonFileManager {
    private:
        // Inctance variables
        nlohmann::json json;
        std::vector<std::tuple<double, double, double>> joints;

        // Private Methods
        void parseJSON(std::string file); // Parse jason data
        void estimatedJoints(); // compute joint locations and put to joints variable

    public:
        //Getter
        nlohmann::json getJSON(){return json;};
        std::vector<std::tuple<double, double, double>> getJoints(){return joints;};

        // Methods
        JsonFileManager(std::string file); // Constructor
        int jsonFileManagerDidLoad(int argc, char**argv);
        //void jsonPrint();
};

//#endif