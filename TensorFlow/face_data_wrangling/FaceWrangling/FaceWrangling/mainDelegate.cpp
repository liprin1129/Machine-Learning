//
//  mainDelegate.cpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/10.
//  Copyright © 2018 170. All rights reserved.
//

#include "mainDelegate.hpp"

auto MainDelegate::mainDelegation(int argc, char** argv) -> int{
	
	/*FileSystemManager fsm;
	fsm.fileSystemManagerHasLoaded(argc, argv[1]);*/
	
	DlibFaceDetector dlibFD;
	dlibFD.dlibFaceDetectorHasLoaded(argc, argv[1]);
	return 0;
}
