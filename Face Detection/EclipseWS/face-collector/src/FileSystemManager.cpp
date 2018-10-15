/*
 * FileSystemManager.cpp
 *
 *  Created on: Jun 6, 2018
 *      Author: user170
 */

#include "FileSystemManager.hpp"

FileSystemManager::FileSystemManager() {
	//this->_absDirName = absDirName;
	//std::cout << "Enter your name: " << std::endl;
	/*
	bool dir;
	dir = this->makeDir("/home/user170/Desktop/test");
	std::cout << dir << ": Create a directory." << std::endl;
	*/
}

/*
FileSystemManager::~FileSystemManager() {
	// TODO Auto-generated destructor stub
}*/

bool FileSystemManager::makeDir(std::string absDirName) {
	this->_absDirName = absDirName;
	// boost::filesystem::exists
	return boostFS::create_directories(absDirName);
}

void FileSystemManager::saveFaceImage(cv::Mat faceCvMat) {
	int fileCount = this->numOfFiles(this->_absDirName);

	cv::imwrite(this->_absDirName+boost::lexical_cast<std::string>(fileCount)+".png", faceCvMat);
}

int FileSystemManager::numOfFiles(std::string absDirName) {
	//boostFS::directory_iterator dir(absDirName);
	int fileCount = 0;

	for(boostFS::directory_iterator iter(absDirName), iter_end; iter!=iter_end; ++iter){
		if (boostFS::is_regular_file(iter->status()))
			fileCount += 1;
	}
	//std::cout << dir.filename() << '\n';
	//std::cout << "Number of files: " << fileCount << '\n';

	return fileCount;
}
