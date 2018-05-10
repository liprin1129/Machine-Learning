//
//  fileSystemManager.cpp
//  FaceWrangling
//
//  Created by 170 on 2018/05/10.
//  Copyright © 2018 170. All rights reserved.
//

#include "fileSystemManager.hpp"

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

auto FileSystemManager::isDirectory(std::string dir_path) -> int {
	boost::filesystem::recursive_directory_iterator dir(dir_path), end;
	while (dir != end)
	{
		if (boost::filesystem::is_directory(dir->status())){
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
		return 0;
}

auto FileSystemManager::isFile(std::string dir_path) -> int {
	boost::filesystem::recursive_directory_iterator dir(dir_path), end;
	while (dir != end)
	{
		if (boost::filesystem::is_regular_file(dir->status())){
			//std::cout << dir->path().parent_path();
			//std::cout << dir->path().filename() << "\n";
			std::cout << dir->path().parent_path().string() + dir->path().filename().string() << std::endl;
		}
		++dir;
	}
	return 0;
}
auto FileSystemManager::classHasLoaded(int argc, ...) -> int{
//auto FileSystemManager::classHasLoaded(int argc, char** argv) -> int{
	
	// Set cstdarg class for managain multiple variables
	va_list argv;
	va_start(argv, argc);
	
	if(argc > 1){
		auto i = va_arg(argv, char*);
		
		//std::cout << i << std::endl;
		
		//this->isDirectory(i);
		this->isFile(i);
	}
	
	va_end(argv);
	
	return 0;
}
