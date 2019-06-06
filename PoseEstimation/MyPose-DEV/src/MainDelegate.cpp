#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
	// argv[1]: json file
	
	CameraManager cm;
	//return cm.cameraManagerDidLoad(argc, argv);

	// JSON test
    JsonFileManager jfm(argv[1], argv[2]);
    //jfm.jsonPrint();
    jfm.jsonFileManagerDidLoad();

	
	//auto joints = jfm.getJoints();
	//fprintf(stdout, "Keypoints vector size: %d\n", (int)joints.size());
	JointCoordinateManager jcm(jfm.getLeftJoints(), jfm.getRightJoints());
	return jcm.jointCoordinateManagerDidLoad(cm.getFocalLength());
}