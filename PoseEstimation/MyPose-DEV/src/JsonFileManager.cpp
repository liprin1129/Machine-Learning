#include "JsonFileManager.h"

//using json = nlohmann::json;
JsonFileManager::JsonFileManager(std::string file) {
    //parseJSON("left_keypoints.json");
    parseJSON(file);
    //std::cout << json["people"][0]["pose_keypoints_2d"] << std::endl;
}


void JsonFileManager::parseJSON(std::string file) {
    std::ifstream reading(file, std::ios::in);
    //auto reading = std::ifstream(file, std::ios::in);
    
    // json instance
    //nlohmann::json json;
    reading >> json;
}


void JsonFileManager::estimatedJoints() {
    //std::vector<std::tuple<double, double, double>> joints;
    std::tuple<double, double, double> joint;
    
    auto keypoints = json["people"][0]["pose_keypoints_2d"];
    auto epsilon = 0.00001; // float number for comparing kypoints whether they are not zero

    for (int i=0; i<keypoints.size()/3; ++i) {        
        bool truefalse = false;
        // Check all points are not zero
        if (((double)keypoints[i*3] > epsilon) && ((double)keypoints[i*3+1] > epsilon) && ((double)keypoints[i*3+2] > epsilon)) {
            joint = std::make_tuple((double)keypoints[i*3], (double)keypoints[i*3+1], (double)keypoints[i*3+2]);
            joints.push_back(joint);
            truefalse = true;
        }
        //fprintf(stdout, "iteration: %d, %lf, %lf, %lf <%d>\n", i, (double)keypoints[i*3], (double)keypoints[i*3+1], (double)keypoints[i*3+2], truefalse);
    }
    fprintf(stdout, "Keypoints vector size: %d\n", (int)joints.size());
}


int JsonFileManager::jsonFileManagerDidLoad(int argc, char**argv) {
    estimatedJoints();
}
/*
void JsonFileManager::jsonPrint() {

    std::ifstream reading("left_keypoints.json", std::ios::in);
    nlohmann::json json;
    reading >> json;
    //std::string content( (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>() );
    //json j = json::parse(content);
    std::cout << json["people"][0]["pose_keypoints_2d"] << std::endl;
}
*/