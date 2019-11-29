#include <iostream>
#include <experimental/filesystem>

class DirectoryAndFileManager {

    public:
        DirectoryAndFileManager();
        void lookupDirectory(std::string directoryName, int &numFiles);
};

