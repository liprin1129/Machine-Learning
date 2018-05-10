//
//  mainDelegate.cpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/10.
//  Copyright © 2018 170. All rights reserved.
//

#include "mainDelegate.hpp"

auto MainDelegate::mainDelegation(int argc, char** argv) -> int{
	
	FileSystemManager fsm;
	//fsm.classHasLoaded(2, "s1", "s2");
	fsm.classHasLoaded(argc, argv[1]);
	
	return 0;
}
