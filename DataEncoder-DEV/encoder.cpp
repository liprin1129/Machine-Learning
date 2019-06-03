#include "encoder.h"

Encoder::Encoder() {
    auto randomInputDouble = (Eigen::MatrixXd::Random(3, 2) + Eigen::MatrixXd::Ones(3, 2));
    auto randomInputInt = randomInputDouble.cast<int>();
    _rawData = randomInputInt.cast<double>();
    //std::cout << result << std::endl;
    
    auto keyDouble = (Eigen::MatrixXd::Random(2, 3) + Eigen::MatrixXd::Ones(2, 3));
    auto keyInt = keyDouble.cast<int>();
    _keyW = keyInt.cast<double>();
    //_keyW = Eigen::MatrixXd::Random(_rawData.rows(), _rawData.cols());
}
/*
void Encoder::encryption(Eigen::MatrixXd A, Eigen::MatrixXd W) {
    _encryptedData = W*A;
}

void Encoder::pinv(Eigen::MatrixXd W) {
    _pinvKeyW = W.completeOrthogonalDecomposition().pseudoInverse();
}
*/
void Encoder::encryption() {
    _encryptedData = _keyW*_rawData;
}

double floorX(double &a) {
    return floor(a);
}

void Encoder::pinv() {
    //_pinvKeyW = _keyW.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(_keyW, Eigen::ComputeThinV | Eigen::ComputeThinU);
    std::cout << 'U' << std::endl << svd.matrixU() << std::endl;
    std::cout << 'S' << std::endl << svd.singularValues() << std::endl;
    std::cout << 'V' << std::endl << svd.matrixV() << std::endl;
    _pinvKeyW = svd.matrixV()*svd.singularValues().transpose()*svd.matrixU().transpose();
}

void Encoder::decryption() {
    _decryptedData = _pinvKeyW*_encryptedData;
}

void Encoder::printResult() {
    std::cout << _rawData << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << _keyW << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << _encryptedData << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << _decryptedData << std::endl;
    std::cout << "----------------" << std::endl;
    auto result = Eigen::MatrixXd(_decryptedData.rows(), _decryptedData.cols());
    for (int i=0; i<_decryptedData.rows(); i++) {
        for (int j=0; j<_decryptedData.cols(); j++) {
            result(i, j) = round(abs(_decryptedData(i, j)));
            //std::cout << floor(abs(_decryptedData(i, j))) << ' ' << std::endl;
        }
    }
    std::cout << result << std::endl;
}

int main(int argc, char** argv) {
    Encoder encoderInstance;
    encoderInstance.encryption();
    encoderInstance.pinv();
    //encoderInstance.decryption();
    encoderInstance.printResult();
}