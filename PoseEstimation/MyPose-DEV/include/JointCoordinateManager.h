#ifndef __JOINT_COORDINAT_EMANAGER_H__
#define __JOINTCOORDINATEMANAGER_H__

#include <fstream>
#include <tuple>
#include <vector>

class JointCoordinateManager {
    private:
        std::vector<std::tuple<double, double, double>> keyPointCoordinates; // Coordinates output of KyePoints from the left and right images
        std::vector<std::tuple<double, double, double>> leftJoints; // vector having the left image's keypoints
        std::vector<std::tuple<double, double, double>> rightJoints; // vector having the right image's keypoints

    public:
        // Getters
        std::vector<std::tuple<double, double, double>> getkeyPointCoordinates(){return keyPointCoordinates;};

        JointCoordinateManager(
            std::vector<std::tuple<double, double, double>> lj, 
            std::vector<std::tuple<double, double, double>> rj); // Constructor
        int estimateDepthCoordinates(int focalLength); // Compute Z coordinate (This is not a good algorithm. This is only for simple verification.)
        
        int jointCoordinateManagerDidLoad(int focalLength);
};
#endif /* __JOINTCOORDINATEMANAGER_H__ */