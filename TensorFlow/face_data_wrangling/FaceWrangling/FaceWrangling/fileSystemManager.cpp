//
//  fileSystemManager.cpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/10.
//  Copyright © 2018 170. All rights reserved.
//

#include "fileSystemManager.hpp"

namespace boostFS = boost::filesystem;

void commentOut(){
/*
auto FileSystemManager::isDirectory(std::string dir_path) -> int{
	boost::filesystem::path path(p);
	
	if (boost::filesystem::exists(path))    // does p actually exist?
	{
		if (boost::filesystem::is_regular_file(path))        // is p a regular file?
			std::cout << path << " size is " << boost::filesystem::file_size(path) << '\n';
		
		else if (boost::filesystem::is_directory(path)){      // is p a directory?
			std::cout << path << "is a directory\n";
			copy(boost::filesystem::directory_iterator(p), boost::filesystem::directory_iterator(),  // directory_iterator::value_type
				 std::ostream_iterator<boost::filesystem::directory_entry>(std::cout, "\n"));  // is directory_entry, which is
			// converted to a path by the
			// path stream inserter
		}
		else{
			std::cout << path << "exists, but is neither a regular file nor a directory\n";
		}
			return 0;
	}
	else{
		std::cout << path << "does not exist\n";
		return -1;
	}
}
*/

/*
std::vector<std::string> FileSystemManager::isDirectory(std::string dir_path){
	std::vector<std::string> allDir;
	
	if (boost::filesystem::exists(dir_path)) { // does p actually exist?
		boostFS::recursive_directory_iterator dir(dir_path), end;
		while (dir != end)
		{
			if (boostFS::is_directory(dir->status())){
				std::cout << dir->path().parent_path()<< std::endl;
				if (dir.level() == 0)
					std::cout << dir->path().filename() <<  " | " << dir.level() << "\n";
				else{
					std::string whitespace;
					for (int i=0; i<dir.level(); i++){
						whitespace += "     ";
					}
					std::cout << whitespace << dir->path().filename() <<  " | " << dir.level() << "\n";
				}
			}
			++dir;
		}
	}
	else {
		std::error_code ec (0, std::generic_category());
		
		std::error_condition ok;
		if (ec != ok) std::cout << "Custom Error: " << ec.message() << std::endl;
	}
}
*/
}

template<typename T1, typename T2>
void FileSystemManager::fileInvestigator(T1& dir_path, T2& ref_extension){
	if (boostFS::exists(dir_path)) { // does p actually exist?
		boostFS::recursive_directory_iterator dir(dir_path), end;
		while (dir != end)
		{
			if (boostFS::is_regular_file(dir->status())){
				//std::cout << dir->path().parent_path();
				//std::cout << dir->path().filename() << "\n";
				//std::cout << dir->path().parent_path().string() + dir->path().filename().string() << std::endl;
				
				auto extension = boostFS::extension(dir->path().parent_path().string() + '/' + dir->path().filename().string());
				//std::cout << extension << std::endl;
				
				if (extension == ref_extension)
					_allFileAbsPath.push_back(dir->path().parent_path().string() + '/' + dir->path().filename().string());
			}
			++dir;
		}
	}
	else {
		std::cout << "NO!" <<std::endl;
		std::error_code ec (0, std::generic_category());
		
		std::error_condition ok;
		if (ec == ok) std::cout << "Custom Error: " << ec.message() << std::endl;
	}
}

template <typename T>
void FileSystemManager::saveFile(std::string fileName, T& fileData){
    std::ofstream writeFile;
    writeFile.open(fileName);
	
	auto a = FileSystemManager::readFile(fileName);
}

std::tuple<long, std::ifstream> FileSystemManager::readFile(std::string fileName){
	std::ifstream infile(fileName, std::ifstream::binary);
	
	// Get size of file
	infile.seekg(0, infile.end);
	long size = infile.tellg();
	infile.seekg(0);
	
	std::cout << "File size:" << size << std::endl;
	
	auto data = std::make_tuple(size, infile);
	
	return data;
}

int FileSystemManager::fileSystemManagerHasLoaded(int argc, ...){
//auto FileSystemManager::classHasLoaded(int argc, char** argv) -> int{
	
	// Set cstdarg class for managain multiple variables
	va_list argv;
	va_start(argv, argc);
	
	if(argc > 1){
		auto dirPath = va_arg(argv, char*);
		
		//std::cout << i << std::endl;
		
		//this->isDirectory(i);
		this->fileInvestigator(dirPath, ".jpg");
		std::cout << _allFileAbsPath.size() << std::endl;
		/*
		for (std::vector<std::string>::const_iterator iter=fileAbsPath.begin(); iter!=fileAbsPath.end(); ++iter){
			std::cout << *iter << std::endl;
		}*/
			 //fileAbsPath.begin(), fileAbsPath.end(), boost::bind<std::string>(printf, "%s\n", _1));
		this->saveFile("/Users/user170/Developments/Personal-Dev./Machine-Learning/Data/Face/lfw/German_Khan/German_Khan_0001.jpg", dirPath);
	}
	
	va_end(argv);
	
	return 0;
}