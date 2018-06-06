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
public:
	FileSystemManager();
	//virtual ~FileSystemManager();

	bool makeDir(std::string absDirName);
};

#endif /* FILESYSTEMMANAGER_HPP_ */
