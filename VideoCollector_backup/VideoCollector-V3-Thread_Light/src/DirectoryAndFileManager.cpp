#include "DirectoryAndFileManager.h"

DirectoryAndFileManager::DirectoryAndFileManager() {};

void DirectoryAndFileManager::lookupDirectory(std::string directoryName, int &numFiles) {
    for (auto &p: std::experimental::filesystem::directory_iterator(directoryName)) {
        //std::cout << p.path() << std::endl;
        ++numFiles;
    }
    //std::cout << numFiles << std::endl;
}