//
//  fileSystemManager.hpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/10.
//  Copyright © 2018 170. All rights reserved.
//

#ifndef fileSystemManager_hpp
#define fileSystemManager_hpp

//#include <stdio.h>
#include "commonHeader.h"

class FileSystemManager {
	
public:
	auto isDirectory(std::string dir_path) -> int;
	
	auto classHasLoaded(int argc, ...) -> int;
	auto isFile(std::string dir_path) -> int;
	//auto classHasLoaded(int argc, char** argv) -> int;
};
#endif /* fileSystemManager_hpp */
