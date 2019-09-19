#ifndef __JSON_FILE_MANAGER_H__
#define __JSON_FILE_MANAGER_H__

#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <openpose/pose/poseParameters.hpp>

class JsonFileManager {
    private:
        // Inctance variables
        nlohmann::json leftJSON;
        nlohmann::json rightJSON;
        std::vector<std::tuple<double, double, double>> leftJoints;
        std::vector<std::tuple<double, double, double>> rightJoints;

        // Private Methods
        nlohmann::json parseJSON(std::string file); // Parse jason data
        std::vector<std::tuple<double, double, double>> estimatedJoints(nlohmann::json json); // compute joint locations and put to joints variable

    public:
        //Getter
        nlohmann::json getLeftJSON(){return leftJSON;};
        nlohmann::json getRightJSON(){return rightJSON;};
        std::vector<std::tuple<double, double, double>> getLeftJoints(){return leftJoints;};
        std::vector<std::tuple<double, double, double>> getRightJoints(){return rightJoints;};

        // Methods
        JsonFileManager(std::string lF, std::string rF); // Constructor
        void writeJSON2File(std::string filename, std::vector<std::tuple<double, double, double>> keyPointCoordinates);
        void jsonFileManagerDidLoad();
        //void jsonPrint();
};

#endif /* __JSON_FILE_MANAGER_H__ */