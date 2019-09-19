#include "JointCoordinateManager.h"

JointCoordinateManager::JointCoordinateManager(
    std::vector<std::tuple<double, double, double>> lJ, 
    std::vector<std::tuple<double, double, double>> rJ) {
// Constructor
    leftJoints = lJ;
    rightJoints = rJ;
}


int JointCoordinateManager::estimateDepthCoordinates(int focalLength) {
// Compute Z coordinate (This is not a good algorithm. This is only for simple verification.)

    if ((int)leftJoints.size() == (int)rightJoints.size()) {

        for (int i=0; i<(int)leftJoints.size(); ++i) {
            auto lJ = leftJoints[i]; // get a tuple from the leftJoints vector
            auto rJ = rightJoints[i]; // get a tuple from the rightJoints vector

            auto [lX, lY, lC] = lJ; // untie the tuple for left image, x, y and confidence
            auto [rX, rY, rC] = rJ; // untime the tuple for right image
            //fprintf(stdout, "%lf:%lf, %lf:%lf, %lf:%lf\n", lX, rX, lY, rY, lC, rC);

            // TEMPORAL CODE FOR SIMPLE ALGORITHM
            auto Z = static_cast<double>(120)*focalLength/(rX-lX); // compute Z coordinate, 120 is baseline in cm
            //auto X = (lX - cx)*Z/fx;
            //auto Y = (lY - cy)*Z/fy;

            auto joint = std::make_tuple(rX, rY, Z); // temporally save coordinates
            keyPointCoordinates.push_back(joint);

            //fprintf(stdout, "Joint Coordinate %d: [%lf, %lf, %lf]\n", i, lX, lY, Z);
        }
        return 0;
    }
    else {
        fprintf(stdout, "ERROR: The numbers of elements in left and right joint vector are not same.");
        return -1;
    }
}


int JointCoordinateManager::jointCoordinateManagerDidLoad(int focalLength) {
    if (estimateDepthCoordinates(focalLength) != 0) {
        return -1;
    }

    return 0;
}