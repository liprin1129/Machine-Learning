#include "CustomDataLoader.h"

CustomDataset::CustomDataset(const std::string& locCSV, const std::string& locImages, std::tuple<int, int> newSize, bool verbose) {
    _locCSV = locCSV;
    _locImages = locImages;
    
    readCSV(locCSV);

    _rescale = newSize;

    _verbose = verbose;
}

torch::data::Example<> CustomDataset::get(size_t index)
{
    //if (_verbose) std::fprintf(stdout, "CustomDataset get called\n");

    auto [imgName, label] = _dataset[index];

    std::string imgPath = _locImages;
    imgPath += imgName;
    
    //std::cout << imgPath << std::endl;

    // Load image with OpenCV.
    cv::Mat img = cv::imread(imgPath);
    //if (_verbose) checkcvMatNan(img, "CV_8UC3");

    img.convertTo(img, CV_32FC3); // Convert CV_8UC3 data type to CV_32FC3
    //if (_verbose) checkcvMatNan(img, "CV_32FC3");

    // Rescale
    /*auto rescale = Rescale(_rescale);
    rescale(img, label);
    auto [rImg, rLabel] = rescale.getResizedDataCVandFloat();
    //img = rImg/255; // rescale to [0, 1]
    img = rImg;
    */
    //auto rLabel = label;
    //std::cout << rLabel << std::endl;

    // Convert the image and label to a tensor.
    torch::TensorOptions imgOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, imgOptions);
    imgTensor = imgTensor.permute({2, 0, 1}); // convert to CxHxW
    //if (_verbose) printf("nan in the img Tensor: %s\n", torch::isnan(imgTensor).sum().item<int>() ? "nan detected" : "nan not detected");
    //std::cout << imgTensor.sizes() << std::endl;

    // Convert int label to a tensor
    float labelsArr[label.size()];
    std::copy(label.begin(), label.end(), labelsArr);

    torch::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    //torch::TensorOptions labelOptions = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor labelTensor = torch::from_blob(labelsArr, {1, (signed long) label.size()}, labelOptions);
    //labelTensor = labelTensor.div(std::get<0>(_rescale));
    //if (_verbose) printf("nan in the label Tensor: %s\n", torch::isnan(labelTensor).sum().item<int>() ? "nan detected" : "nan not detected");

    //checkTensorImgAndLandmarksV2(img, label, imgTensor, labelTensor);
    //checkTensorImgAndLandmarksV1(std::move(imgTensor.clone()), std::move(labelTensor.clone()));

    return {imgTensor.clone(), labelTensor.clone()};
}

void CustomDataset::checkcvMatNan(cv::Mat img, std::string dTypeStr) {
    cv::Mat rgbChannel[3];
    split(img, rgbChannel);

    int numNan = 0;
    int totNum = 0;
    for (auto ch: {0, 1, 2}) {
        //std::fprintf(stdout, "ch: %d\n", ch);
        for (int col=0; col<img.cols; ++col){
            //std::fprintf(stdout, "col: %d, ", col);
            for (int row=0; row<img.rows; ++row){
                //std::fprintf(stdout, "row: %d, ", row);
                if (rgbChannel[ch].at<int>(row, col) < 0 and rgbChannel[ch].at<int>(row, col) > 254){
                    numNan += 1;
                }
                totNum += 1;
            }   
        }
    }
    std::fprintf(stdout, "None zero in the img (%s): %d / %d\n", dTypeStr.c_str(), numNan, totNum);

    //int numNan = 0;
    //for (auto channel: rgbChannel) {
    //    numNan += cv::countNonZero(channel);
    //}
    //std::fprintf(stdout, "None zero in the img (scailing): %d / %d\n", numNan/3, img.cols*img.rows);
}

void CustomDataset::readCSV(const std::string &loc) {
    //std::vector<std::tuple<std::string, std::vector<float>>> dataset;

    // File pointer
    std::fstream fin;

    // Open an existing record 
    fin.open(loc, std::fstream::in); 

    // Read the Data from the file as String Vector and convert it to int
    std::vector<float> label;

    std::string temp, imgName, coord; 

    getline(fin, temp); // Skip the frist row

    while (fin >> temp) {
        label.clear();

        std::stringstream s(temp);

        std::getline(s, imgName, ',');
        //if (_verbose) std::cout << imgName << std::endl;
        
        while (std::getline(s, coord, ',')) {
            //label.push_back(stoi(coord));
            //std::cout << stof(coord) << ", ";
            label.push_back(stof(coord));
            //if (_verbose) std::cout << coord << ", ";
        }
        //if(_verbose) std::cout << std::endl;
        
        _dataset.push_back(std::make_tuple(imgName, label));
    }
    //std::cout << std::get<1>(_dataset[0]) << std::endl;
}