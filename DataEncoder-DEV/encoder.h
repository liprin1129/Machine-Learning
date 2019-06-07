#include <iostream>
#include <Eigen/Dense>

class Encoder {
    private:
        Eigen::MatrixXd _data; // raw input data
        Eigen::Matrix4d _keyW; // weight matrix W for key
        Eigen::MatrixXd _encryptedData; // encryptedData
        Eigen::Matrix4d _invKeyW; // inverse matrix of W
        Eigen::MatrixXd _pinvKeyW; // pseudoinverse matrix of W
        Eigen::MatrixXd _decryptedData; // decrypted data
        // METHODS

    public:
        Encoder(); // constructor
        //void encryption(Eigen::MatrixXd A, Eigen::MatrixXd W);
        //void pinv(Eigen::MatrixXd W);
        void encryption();
        Eigen::Matrix4d keyInv(Eigen::Matrix4d data);
        void pinv();
        void printResult();
        void decryption();
};