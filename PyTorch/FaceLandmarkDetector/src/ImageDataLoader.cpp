#include "ImageDataLoader.h"

void ImageDataLoader::readLabel() {
    /*
    std::ifstream ptsFile("300W/Indoor/indoor_001.pts");
    //std::ifstream ptsFile ("./indoor_001.pts");
    std::string line;

    if (ptsFile.is_open())
    {
        while ( getline (ptsFile,line) )
        {
        std::cout << line << std::endl;
        }
        ptsFile.close();
    } else {
        //std::cout << "Fail to Load\n" << system("ls");
        std::cerr << "Failed to open directory. Exiting out..." << std::endl;
        exit(-1);
    }
    */
   auto path = "/DATASETs/Face/Landmarks/300W/";

  for(auto& p: std::filesystem::directory_iterator(path))
        std::cout << p.path() << '\n';
}