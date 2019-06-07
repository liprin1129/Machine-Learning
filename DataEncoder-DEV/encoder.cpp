#include "encoder.h"

Encoder::Encoder() {

    auto keyD = (Eigen::Matrix4d::Random(4, 4) + Eigen::Matrix4d::Ones(4, 4));
    auto keyI = keyD.cast<int>();
    _keyW = keyI.cast<double>();
    //std::cout << result << std::endl;
    /*
    _rawData = Eigen::MatrixXd(8, 8);
    Eigen::VectorXd v(8); v << 1, 0, 0, 1, 1, 1, 0, 1;
    //Eigen::VectorXd v(8); v << 0, 1, 0, 0, 1, 1, 1, 0;
    for (int i=0; i<8; ++i) {
        _rawData.row(i) = v;
    }
    */
    auto dataD = (Eigen::MatrixXd::Random(4, 2) + Eigen::MatrixXd::Ones(4, 2));
    auto dataI = dataD.cast<int>();
    _data = dataI.cast<double>();
    //_keyW = Eigen::MatrixXd::Random(_rawData.rows(), _rawData.cols());

    std::cout << _keyW.inverse() << std::endl;
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
    _encryptedData = _keyW*_data;
}

double floorX(double &a) {
    return floor(a);
}

Eigen::Matrix4d Encoder::keyInv(Eigen::Matrix4d data) {
    std::cout << data.inverse() << std::endl;
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
    std::cout << _data << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << _keyW << std::endl;
    std::cout << "----------------" << std::endl;
    //std::cout << _encryptedData << std::endl;
    std::cout << "----------------" << std::endl;
    //std::cout << keyInv(_keyW) << std::endl;
    /*
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
    */
}

int main(int argc, char** argv) {
    Encoder encoder;

    encoder.encryption();
    encoder.printResult();
}