#include "DataLoader.h"

DataLoader::DataLoader(std::string path) {
    readDataDirectory(path);
}


void DataLoader::readDataDirectory(std::string rootPath) {
    //auto path = "/DATASETs/Face/Landmarks/300W/";

    std::string pathName;                                                                                                               // Temporal variable for absolute directory path

    for(auto& fileName: filesystem::recursive_directory_iterator(rootPath)) {
        if (filesystem::is_directory(fileName)) pathName = fileName.path();                                                             // if fileName is a directory, save absolute directory path
    
        if (filesystem::is_regular_file(fileName)) {                                                                                    // if fileName is a file
            auto fileNameStr = (std::string) fileName.path().filename();                                                                // extract filename from absolute directory path
            /*
            if (fileNameStr.find("png")!=std::string::npos && fileNameStr.find("indoor") !=std::string::npos) {                         // if filename includes "png" and "indoor" string
                std::string tmpNameStr = fileNameStr;                                                                                   // copy filename string to tmpNameStr 
                _dataset.push_back(std::make_tuple(pathName + "/" + fileNameStr, pathName + "/" + tmpNameStr.replace(11, 3, "pts")));    
            }
            else if (fileNameStr.find("png")!=std::string::npos && fileNameStr.find("outdoor") !=std::string::npos) {
                std::string tmpNameStr = fileNameStr;
                _dataset.push_back(std::make_tuple(pathName + "/" + fileNameStr, pathName + "/" + tmpNameStr.replace(12, 3, "pts")));
            }
            */
           if (fileNameStr.find("png")!=std::string::npos) {
               std::string tmpNameStr = fileNameStr;
               _dataset.push_back(std::make_tuple(pathName + "/" + fileNameStr, pathName + "/" + tmpNameStr.replace(tmpNameStr.end()-3, tmpNameStr.end(), "pts")));    
           }
        }
    }

    //std::cout << _dataset.size() << std::endl;
    /*
    // Print dataset vector
    for (auto &data : _dataset) {
        auto [image, label] = data;
        std::cout << image << " | " << label << std::endl;
    }
    */
}


cv::Mat DataLoader::readImage2CVMat(std::string filePath, bool norm) {
    std::cout << filePath << std::endl;
    
    cv::Mat origImage, normImage;
    
    origImage = cv::imread(filePath, cv::IMREAD_COLOR);

    if (origImage.empty()) {
        std::cerr << "Could not open the image: " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }

    if (norm) {
        cv::normalize(origImage, normImage, 1, 0, cv::NORM_MINMAX, CV_32FC3);
        //std::cout << normImage << std::endl;
        return normImage;
    }
    else {
        return origImage;
    }
    /*
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    imshow("Image", normImage);

    cv::waitKey(0);
    */
}


std::tuple<float, float> DataLoader::labelNormalizer(int col, int row, float X, float Y) {
    X /= col; // (X - min(X)) / (max(X) - min(X))
    Y /= row; // (Y - min(Y)) / (max(Y) - min(Y))

    return std::make_tuple(X, Y);
}


void DataLoader::labelStr2Float(std::tuple<std::string, std::string> filePath, bool norm){
// Convert string type of landmark points to float number list, _labels,
// and at the same time, load an corresponding image to cv::Mat, _image.
// Arguments:
//      filePath: first emelement is a image file absolute path, and second is pst file absolute path
//          type: std::tuple<std::string, std::string>,
//          value: a element of _dataset vector
//      norm: whether normalise or not
//          type: bool
//          value: (default) true

    // Temporal variables
    std::fstream ptsFile; // fstream instach
    
    float X; // point's X coordinat
    float Y; // point's Y coordinat
    //std::tuple<float, float> norm; // tuple for normalized X, Y coordinate

    std::string line; // string to save a line
    std::string::size_type firstDigitSize; // size_t to save the end position of detected digit (in this case, first digit has 6 char xxx.xxx, firstDigitSize will be 7)

    //std::vector<std::tuple<float, float>> landmarks; // output vector containing landmark positions
    std::list<float> landmarks;

    auto [imgPath, landmarksPath] = filePath;

    ptsFile.open(landmarksPath, std::fstream::in); // open fstream with read only mode

    // Tasks
    if (ptsFile.is_open()) {
        _image = readImage2CVMat(imgPath, norm); // to get cols and rows of an image for normalization

        std::getline(ptsFile, line); // skip a line
        std::getline(ptsFile, line); // skip a line
        std::getline(ptsFile, line); // skip a line

        while (std::getline(ptsFile, line)) {
            if (std::isdigit(line[0])) { // Check string is digit?
                X = std::stof(line, &firstDigitSize); // convert first string to float
                Y = std::stof(line.substr(firstDigitSize)); // convert second string to float

                if (norm) {
                    auto [normX, normY] = labelNormalizer(_image.cols, _image.rows, X, Y); // Normalization
                    landmarks.push_back(normX); // push normalized X coordinate to the landmarks vector
                    landmarks.push_back(normY); // push normalized Y coordinate to the landmarks vector
                }

                else {
                    landmarks.push_back(X);
                    landmarks.push_back(Y);
                }
            }
        }
    }

    /*// print landmarks vector
    for (auto &l: landmarks) {
        //auto [X, Y] = l;
        //std::cout << X << ", " << Y << std::endl;
        std::cout << l << std::endl;
    }
    */

    _labels = landmarks;
}


std::tuple<cv::Mat, std::list<float>> DataLoader::loadOneTraninImageAndLabel(std::tuple<std::string, std::string> filePath, bool norm) {
    //auto [imgPath, landmarksPath] = filePath;
    labelStr2Float(filePath, norm);

    return std::make_tuple(_image, _labels);
}