#include "DataLoader.h"

CustomDataset::CustomDataset(const std::string& loc_states) {
    readCSV(loc_states);
}

void checkTensorImgAndLandmarksV1(torch::Tensor imgTensor, torch::Tensor labelTensor) {
    // Convert the image Tensor to cv::Mat with CV_8UC3 data type
    int cvMatSize[2] = {(int)imgTensor.size(1), (int)imgTensor.size(2)};
    cv::Mat imgCV(2, cvMatSize, CV_32FC3, imgTensor.data_ptr());
    imgCV.convertTo(imgCV, CV_8UC3);

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;
    for (int i=0; i<labelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = labelTensor[0][i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 2, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = labelTensor[0][i].item<float>();//*outputImg.cols;
    }
    
    cv::namedWindow("Restored", CV_WINDOW_AUTOSIZE);
    imshow("Restored", imgCV);
    cv::waitKey(0);
}

void checkTensorImgAndLandmarksV2(cv::Mat img, std::vector<int> label, torch::Tensor const &imgTensor, torch::Tensor const &labelTensor) {
    img.convertTo(img, CV_8UC3);

    int origX = 0.0, origY=0.0;
    for (int i=0; i<labelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            origY = label[i];
            cv::circle(img, cv::Point2d(cv::Size(origX, origY)), 2, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        origX = label[i];
    }

    cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);
    imshow("Original", img);
    cv::waitKey(0);

    // Convert the image Tensor to cv::Mat with CV_8UC3 data type
    int cvMatSize[2] = {(int)imgTensor.size(1), (int)imgTensor.size(2)};
    cv::Mat imgCV(2, cvMatSize, CV_32FC3, imgTensor.data_ptr());
    imgCV.convertTo(imgCV, CV_8UC3);

    // Convert the label Tensor to vector
    std::vector<std::tuple<float, float>> landmarks;
    float X = 0.0, Y=0.0;
    for (int i=0; i<labelTensor.size(1); ++i) {
        if (i % 2 == 1) {
            Y = labelTensor[0][i].item<float>();//*outputImg.rows;
            landmarks.push_back(std::make_tuple(X, Y));
            cv::circle(imgCV, cv::Point2d(cv::Size((int)X, (int)Y)), 2, cv::Scalar( 0, 0, 255 ), cv::FILLED, cv::LINE_8);
            //std::cout << (int)X << ", " << (int)Y << std::endl;
        }
        X = labelTensor[0][i].item<float>();//*outputImg.cols;
    }
    
    cv::namedWindow("Restored", CV_WINDOW_AUTOSIZE);
    imshow("Restored", imgCV);
    cv::waitKey(0);
}

torch::data::Example<> CustomDataset::get(size_t index)
{
    auto [imgName, label] = _dataset[index];

    std::string imgPath = "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/";
    imgPath += imgName;

    // Load image with OpenCV.
    cv::Mat img = cv::imread(imgPath);
    img.convertTo(img, CV_32FC3); // Convert CV_8UC3 data type to CV_32FC3

    // Convert the image and label to a tensor.
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
    imgTensor = imgTensor.permute({2, 0, 1}); // convert to CxHxW
    
    // Convert int label to a tensor
    float labelsArr[label.size()];
    std::copy(label.begin(), label.end(), labelsArr);

    torch::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor labelTensor = torch::from_blob(labelsArr, {1, (signed long) label.size()}, labelOptions);

    //checkTensorImgAndLandmarksV2(img, label, imgTensor, labelTensor);
    checkTensorImgAndLandmarksV1(imgTensor.clone(), labelTensor.clone());

    return {imgTensor.clone(), labelTensor.clone()};
}

void CustomDataset::readCSV(const std::string &loc) {
    std::vector<std::tuple<std::string, std::vector<int>>> dataset;

    // File pointer
    std::fstream fin;

    // Open an existing record 
    fin.open(loc, std::fstream::in); 

    // Read the Data from the file as String Vector 
    std::vector<int> label;

    std::string temp, imgName, coord; 

    getline(fin, temp); // Skip the frist row

    while (fin >> temp) {
        label.clear();

        std::stringstream s(temp);

        std::getline(s, imgName, ',');
        //std::cout << imgName << std::endl;
        
        while (std::getline(s, coord, ',')) {
            label.push_back(stoi(coord));
        }
        
        _dataset.push_back(std::make_tuple(imgName, label));
    }
    /*
    while (fin >> temp) {
        row.clear();
        // read an entire row and store it in a string variable 'line' 
        getline(fin, line);
        // used for breaking words 
        std::stringstream s(line);
        
        // read every column data of a row and store it in a string variable, 'coord' 
        while (std::getline(s, coord, ',')) {
            // add all the column data of a row to a vector 
            std::cout << coord << std::endl;
            //row.push_back(stoi(coord));
            row.push_back(coord);
        }
    
        labels.push_back(row);
    }
    */
}