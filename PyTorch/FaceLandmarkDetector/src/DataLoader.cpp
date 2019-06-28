#include "DataLoader.h"

customDataset::customDataset(const std::string& loc_states) {
    readCSV(loc_states);
}

void showImgAndLandmarks(torch::Tensor &imgTensor, torch::Tensor &labelTensor) {
    //std::cout << imgTensor.size(1) << std::endl;
    //std::vector<int64_t>shape = {3, img.size(1), img.size(2)};

    //cv::Mat imgCV = cv::Mat::eye(imgTensor.size(1), imgTensor.size(2), CV_32FC1);
    //std::memcpy((void*)imgCV.data, imgTensor[0].data_ptr(), sizeof(torch::kFloat32)*imgTensor.numel());
    
    //cv::Mat imgCV(cv::Size(imgTensor.size(1), imgTensor.size(2)), CV_8UC1, imgTensor.data_ptr());
    //cv::Mat imgCV((int)imgTensor.size(1), (int)imgTensor.size(2), CV_8UC1, imgTensor[0]. template data<torch::kByte>());
    //cv::Mat imgCV(cv::Size(imgTensor.size(1)/*128*/, imgTensor.size(2)/*128*/), CV_32FC1, imgTensor[0].data<float>());
    
    torch::Tensor intTensor = imgTensor.toType(torch::kInt8);

    cv::Mat imgCV(intTensor.size(1), intTensor.size(2), CV_8UC1, intTensor[0].data<int>());

    std::cout << intTensor[0].sizes() << std::endl;
    std::cout << imgCV.size << std::endl;
    //imgCV.convertTo(imgCV, CV_8UC1);
    std::cout << imgCV << std::endl;
   /*
   torch::Tensor tensor = torch::zeros({3, 2, 3}, torch::kF32);
   cv::Mat cv_mat = cv::Mat::eye(2, 3, CV_32F);

   std::memcpy(tensor[0].data_ptr(), cv_mat.data, sizeof(float)*tensor.numel());

    std::cout << tensor[0] << std::endl<<std::endl;
   std::cout << cv_mat << std::endl;
   */
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    imshow("Image", imgCV);
    cv::waitKey(0);
}

torch::data::Example<> customDataset::get(size_t index)
{
    auto [imgName, label] = _dataset[index];

    std::string imgPath = "/DATASETs/Face/Landmarks/Pytorch-Tutorial-Landmarks-Dataset/faces/";
    imgPath += imgName;

    // Load image with OpenCV.
    cv::Mat img = cv::imread(imgPath);

    // Convert the image and label to a tensor.
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    imgTensor = imgTensor.permute({2, 0, 1}); // convert to CxHxW

    // Convert int label to a tensor
    float labelsArr[label.size()];
    //std::copy(label.begin(), label.end(), labelsArr);

    torch::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor labelTensor = torch::from_blob(labelsArr, {1, (signed long) label.size()}, labelOptions);

    showImgAndLandmarks(imgTensor, labelTensor);

    return {imgTensor.clone(), labelTensor.clone()};
}

void customDataset::readCSV(const std::string &loc) {
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