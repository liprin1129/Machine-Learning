//
//  mainDelegate.hpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/10.
//  Copyright © 2018 170. All rights reserved.
//

#ifndef mainDelegate_hpp
#define mainDelegate_hpp

#include "fileSystemManager.hpp"
#include "dlibFaceDetector.hpp"

class MainDelegate {
	
public:
	auto mainDelegation(int argc, char** argv) -> int;
};

#endif /* mainDelegate_hpp */
