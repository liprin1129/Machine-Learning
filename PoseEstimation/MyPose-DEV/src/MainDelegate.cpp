#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
	//CameraManager cm;
	// argv[1]: json file
	//return cm.cameraManagerDidLoad(argc, argv);

	// JSON test
    JsonFileManager jfm(argv[1]);
    //jfm.jsonPrint();
    jfm.jsonFileManagerDidLoad(argc, argv);
}