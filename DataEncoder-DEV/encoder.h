#include <iostream>
#include <Eigen/Dense>

class Encoder {
    private:
        Eigen::MatrixXd _rawData; // raw input data
        Eigen::MatrixXd _keyW; // weight matrix W for key
        Eigen::MatrixXd _encryptedData; // encryptedData
        Eigen::MatrixXd _pinvKeyW; // pseudoinverse matrix of W
        Eigen::MatrixXd _decryptedData; // decrypted data
        // METHODS

    public:
        Encoder(); // constructor
        //void encryption(Eigen::MatrixXd A, Eigen::MatrixXd W);
        //void pinv(Eigen::MatrixXd W);
        void encryption();
        void pinv();
        void printResult();
        void decryption();
};