/*
 * FileSystemManager.hpp
 *
 *  Created on: Jun 6, 2018
 *      Author: user170
 */

#ifndef FILESYSTEMMANAGER_HPP_
#define FILESYSTEMMANAGER_HPP_

#include "CommonHeaders.hpp"

namespace boostFS = boost::filesystem;

class FileSystemManager {
private:
	std::string _absDirName;

public:
	FileSystemManager();
	//virtual ~FileSystemManager();

	// Getter of _absDirName
	std::string absDirName() const {return _absDirName;}

	// Create a directory with given path and name
	bool makeDir(std::string absDirName);
};

#endif /* FILESYSTEMMANAGER_HPP_ */
