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
protected:
	std::vector<std::string> _allFileAbsPath;
	
public:
	std::vector<std::string> isDirectory(std::string dir_path);
	
	template<typename T1, typename T2>
	void fileInvestigator(T1& dir_path, T2& file_extension);
	//std::vector<std::string> fileInvestigator(T1& dir_path, T2& file_extension);
	
	int fileSystemManagerHasLoaded(int argc, ...);
	//auto classHasLoaded(int argc, char** argv) -> int;
    
    template <typename T> void saveFile(std::string fileName, T& fileData);
	
	// Read a file and return the file size and binary type file
	//struct readFileReturn{long sizeOfFile; std::ifstream inFile;};
	std::tuple<long, std::ifstream> readFile(std::string fileName);
};
#endif /* fileSystemManager_hpp */
