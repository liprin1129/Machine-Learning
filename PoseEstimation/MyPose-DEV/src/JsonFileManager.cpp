#include "JsonFileManager.h"

//using json = nlohmann::json;

void JsonFileManager::jsonPrint() {

    std::ifstream reading("left_keypoints.json", std::ios::in);
    nlohmann::json json;
    reading >> json;
    //std::string content( (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>() );
    //json j = json::parse(content);
    std::cout << json["people"][4] << std::endl;
}