#include "DataLoader.h"

CustomDataset::CustomDataset(const std::string& locCSV, const std::string& locImages, std::tuple<int, int> newSize) {
    _locCSV = locCSV;
    _locImages = locImages;
    
    readCSV(locCSV);

    //_rescale = std::make_unique<Rescale>(newSize);
}


torch::data::Example<> CustomDataset::get(size_t index)
{
    auto [imgName, label] = _dataset[index];

    std::string imgPath = _locImages;"/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/";
    imgPath += imgName;
    
    // Load image with OpenCV.
    cv::Mat img = cv::imread(imgPath);
    //img.convertTo(img, CV_32FC3); // Convert CV_8UC3 data type to CV_32FC3

    // Rescale
    auto rescale = Rescale(std::make_tuple(300, 300));
    rescale(img, label);
    auto [rImg, rLabel] = rescale.getResizedDataCVandFloat();
    img = rImg;

    // Convert the image and label to a tensor.
    torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kInt8).requires_grad(false);
    torch::Tensor imgTensor = torch::from_blob(rImg.data, {rImg.rows, img.cols, 3}, imgOptions);
    imgTensor = imgTensor.permute({2, 0, 1}); // convert to CxHxW

    // Convert int label to a tensor
    float labelsArr[rLabel.size()];
    std::copy(rLabel.begin(), rLabel.end(), labelsArr);

    torch::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor labelTensor = torch::from_blob(labelsArr, {1, (signed long) rLabel.size()}, labelOptions);

    //checkTensorImgAndLandmarksV2(img, label, imgTensor, labelTensor);
    //checkTensorImgAndLandmarksV1(std::move(imgTensor.clone()), std::move(labelTensor.clone()));

    return {imgTensor.toType(torch::kFloat32).clone(), labelTensor.clone()};
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