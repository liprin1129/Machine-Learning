#include "DataLoader.h"

DataLoader::DataLoader(std::string path) {
    readDataDirectory(path);
    
    resizeFlag = false;
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


cv::Mat DataLoader::readImage2CVMat(std::string filePath) {
    // std::cout << filePath << std::endl;
    
    cv::Mat origImage, normImage, resizedImage;
    
    origImage = cv::imread(filePath, cv::IMREAD_COLOR);

    if (origImage.empty()) {
        std::cerr << "Could not open the image: " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }

    return origImage;
    /*
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    imshow("Image", normImage);

    cv::waitKey(0);
    */
}

cv::Mat DataLoader::resizeCVMat(cv::Mat &cvImg, float scaleFactor) {
    cv::Mat resizedImage;
    
    //std::cout << scaleFactor << " | New Size: " << cv::Size2d((int)(cvImg.cols*scaleFactor), (int)(cvImg.rows*scaleFactor)) << std::endl;
    //cv::resize(cvImg, resizedImage, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    cv::resize(cvImg, resizedImage, cv::Size2d(200, 200), 0, 0, cv::INTER_LINEAR);
    
    return resizedImage;
}


std::tuple<float, float> DataLoader::resizeLabel(int origCol, int origRow, int newCol, int newRow, std::tuple<float, float> origLabel){
    auto [X, Y] = origLabel;
    X *= (newCol/(float)origCol); // (X - min(X)) / (max(X) - min(X))
    Y *= (newRow/(float)origRow); // (Y - min(Y)) / (max(Y) - min(Y))

    return std::make_tuple(X, Y);
}


std::tuple<float, float> DataLoader::labelNormalizer(int col, int row, std::tuple<float, float> origLabel) {
    auto [X, Y] = origLabel;

    X /= col; // (X - min(X)) / (max(X) - min(X))
    Y /= row; // (Y - min(Y)) / (max(Y) - min(Y))

    //std::fprintf(stdout, "(%d, %d) -> (%d, %d) = (%f, %f)\n", origCol, origRow, newCol, newRow, X, Y);

    return std::make_tuple(X, Y);
}

/*
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
    std::fstream ptsFile; // fstream instance
    
    float X = 0.0; // point's X coordinat
    float Y = 0.0; // point's Y coordinat
    //std::tuple<float, float> norm; // tuple for normalized X, Y coordinate

    std::string line; // string to save a line
    std::string::size_type firstDigitSize; // size_t to save the end position of detected digit (in this case, first digit has 6 char xxx.xxx, firstDigitSize will be 7)

    //std::vector<std::tuple<float, float>> landmarks; // output vector containing landmark positions
    std::list<float> landmarks;

    auto [imgPath, landmarksPath] = filePath;

    ptsFile.open(landmarksPath, std::fstream::in); // open fstream with read only mode

    // Tasks
    if (ptsFile.is_open()) {
        _image = readImage2CVMat(imgPath, false, norm); // to get cols and rows of an image for normalization

        std::getline(ptsFile, line); // skip a line
        std::getline(ptsFile, line); // skip a line
        std::getline(ptsFile, line); // skip a line

        while (std::getline(ptsFile, line)) {
            if (std::isdigit(line[0])) { // Check string is digit?
                X = std::stof(line, &firstDigitSize); // convert first string to float
                Y = std::stof(line.substr(firstDigitSize)); // convert second string to float
                //std::fprintf(stdout, "\n[Origin] cols: %d, rows: %d | X: %f, Y: %f\n", _image.cols, _image.rows, X, Y);

                if (norm) {
                    auto [normX, normY] = labelNormalizer(_image.cols, _image.rows, X, Y); // Normalization
                    //std::fprintf(stdout, "[Norm] cols: %d, rows: %d | X: %f, Y: %f\n", _image.cols, _image.rows, normX, normY);
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

    // print landmarks vector
    for (auto &l: landmarks) {
        //auto [X, Y] = l;
        //std::cout << X << ", " << Y << std::endl;
        std::cout << l << std::endl;
    }

    _labels = landmarks;
}
*/

std::tuple<float, float> DataLoader::str2Float(std::string strLabel) {
    if (std::isdigit(strLabel[0])) {
        std::string::size_type firstDigitSize;

        float X = std::stof(strLabel, &firstDigitSize);
        float Y = std::stof(strLabel.substr(firstDigitSize));

        return std::make_tuple(X, Y);
    }
    else {
        std::fprintf(stderr, "Label string does not include digits\n");
        exit(-1);
    }
}

std::tuple<cv::Mat, std::list<float>> DataLoader::loadOneTraninImageAndLabel(std::tuple<std::string, std::string> filePath, bool norm) {
    //======================//
    //= FUNCTION VARIABLES =//
    //======================//
    std::fstream ptsFile; // fstream instance
    
    float X = 0.0; // point's X coordinat
    float Y = 0.0; // point's Y coordinat
    //std::tuple<float, float> norm; // tuple for normalized X, Y coordinate

    std::string line; // string to save a line

    auto [imgPath, landmarksPath] = filePath; // Seperate image path and landmarks path

    ptsFile.open(landmarksPath, std::fstream::in); // open fstream with read only mode

    float scaleFactor = 1.0;

    cv::Mat image;
    std::list<float> labels;

    //=================//
    //= FUNCTION BODY =//
    //=================//
    if (ptsFile.is_open()) {
        std::getline(ptsFile, line); // skip a line
        std::getline(ptsFile, line); // skip a line
        std::getline(ptsFile, line); // skip a line

        // Get a image
        image = readImage2CVMat(imgPath); // to get cols and rows of an image for normalization
        int origX = image.cols;
        int origY = image.rows;
        /*
        if (image.cols > 4000 or image.rows > 4000) {
            //std::cout << ">4000!!!" << std::endl;
            scaleFactor = 0.005;
        }
        else if(image.cols > 3000 or image.rows > 3000){
            //std::cout << ">3000!!!" << std::endl;
            scaleFactor = 0.06;
        }
        else if(image.cols > 2000 or image.rows > 2000){
            //std::cout << ">2000!!!" << std::endl;
            scaleFactor = 0.1;
        }
        else if(image.cols > 1000 or image.rows > 1000){
            //std::cout << ">2000!!!" << std::endl;
            scaleFactor = 0.2;
        }
        */
        image = resizeCVMat(image, scaleFactor);

        if (norm) {
            cv::normalize(image, image, 1, 0, cv::NORM_MINMAX, CV_32FC3);
        }

        // Get labels
        while (std::getline(ptsFile, line)) {
            if (std::isdigit(line[0])) { // Check string is digit?
                std::tuple<float, float> label = str2Float(line); // Read X, Y point with float type
                
                label = resizeLabel(origX, origY, image.cols, image.rows, label); // Read X, Y point with float type

                if (norm) {
                    label = labelNormalizer(image.cols, image.rows, label);
                    //std::cout << "Norm Label!!" << std::endl;
                }

                labels.push_back(std::get<0>(label));
                labels.push_back(std::get<1>(label));

                //std::fprintf(stdout, "(%d, %d) -> (%d, %d) = (%f, %f)\n", origX, origY, image.cols, image.rows, std::get<0>(label)*image.cols, std::get<1>(label)*image.rows);
                //cv::circle(image, cv::Point2d(cv::Size(std::get<0>(label)*image.cols, std::get<1>(label)*image.rows)), 5, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            }
        }
    }
    /*
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    imshow("Image", image);

    cv::waitKey(0);
    */
    return std::make_tuple(image, labels);
}