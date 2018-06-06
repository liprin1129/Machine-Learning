/*
 * FileSystemManager.cpp
 *
 *  Created on: Jun 6, 2018
 *      Author: user170
 */

#include "FileSystemManager.hpp"

FileSystemManager::FileSystemManager() {
	bool dir;

	dir = this->makeDir("/home/user170/Desktop/test");
	std::cout << dir << ": Create a directory." << std::endl;
}
/*
FileSystemManager::~FileSystemManager() {
	// TODO Auto-generated destructor stub
}*/

bool FileSystemManager::makeDir(std::string absDirName) {
	return boostFS::create_directories(absDirName);
}
