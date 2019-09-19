#include "JsonFileManager.h"

//using json = nlohmann::json;
JsonFileManager::JsonFileManager(std::string lF, std::string rF) {
    leftJSON = parseJSON(lF);
    rightJSON = parseJSON(rF);
    //std::cout << json["people"][0]["pose_keypoints_2d"] << std::endl;
}


nlohmann::json JsonFileManager::parseJSON(std::string file) {
    nlohmann::json json;
    
    std::ifstream readingLeftJSON(file, std::ios::in);
    //auto reading = std::ifstream(file, std::ios::in);
    
    readingLeftJSON >> json;

    return json;
}


std::vector<std::tuple<double, double, double>> JsonFileManager::estimatedJoints(nlohmann::json json) {
    std::vector<std::tuple<double, double, double>> joints;
    std::tuple<double, double, double> joint;
    
    auto keypoints = json["people"][0]["pose_keypoints_2d"]; // Data through People -> 0 -> pose_keypoints_2d
    auto epsilon = 0.00001; // double number for comparing kypoints whether they are not zero

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
    //fprintf(stdout, "Keypoints vector size: %d\n", (int)joints.size());

    return joints;
}

void JsonFileManager::writeJSON2File(std::string filename, std::vector<std::tuple<double, double, double>> keyPointCoordinates) {
    nlohmann::json jsonObject;

    const auto& poseBodyPartMappingBody25 = op::getPoseBodyPartMapping(op::PoseModel::BODY_25);
    
    for (int i = 0; i < keyPointCoordinates.size(); ++i) {
        auto [X, Y, Z] = keyPointCoordinates[i];
        //std::string s = poseBodyPartMappingBody25.at(i);
        std::fprintf(stdout, "%s: %f, %f, %f\n", poseBodyPartMappingBody25.at(i).c_str(), X, Y, Z);

        jsonObject[poseBodyPartMappingBody25.at(i)] = {X, Y, Z};
    }

    std::ofstream file("key.json");
    file << jsonObject;
    /*
    for (auto i: poseBodyPartMappingBody25) {
        std::fprintf(stdout, "%d: %s\n", i.first, i.second.c_str());
    }
    */
}

void JsonFileManager::jsonFileManagerDidLoad() {
    leftJoints = estimatedJoints(leftJSON);
    rightJoints = estimatedJoints(rightJSON);
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