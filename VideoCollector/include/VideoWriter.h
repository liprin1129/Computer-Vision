#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include <thread>
#include <chrono>
#include <mutex>

class VideoWriter {
    private:
        cv::VideoWriter vw;
        cv::Mat _originLeftCpuMat, _originSideBySideCpuMat;
        std::vector<cv::cuda::GpuMat> _originLeftGpuMat, _originRightGpuMat, _originSideBySideGpuMat;

        int fileCount;
        bool recording, isOpticTriggered;
        std::string toSavePath;

    public:
        VideoWriter();
        
        void writeCvGpuFramesToVideoFormat(
            std::mutex &threadLockMutex, char &key,
            std::string fileName, int numFrames, cv::Size frameSize, 
            std::vector<cv::cuda::GpuMat> &cvLeftGpuMatFrames, std::vector<cv::cuda::GpuMat> &cvRightGpuMatFrames,
            int &numOfFiles, bool &opticalFlowDetectedFlag, bool &isVectorFull);

        void writeSideBySideCvGpuFramesToVideoFormat(
            std::mutex &threadLockMutex, char &key,
            std::string fileName, int numFrames, cv::Size frameSize, 
            std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames,
            int &numOfFiles, bool &opticalFlowDetectedFlag, bool &isVectorFull);

        void writeSlFramesToVideoFormat(
            std::mutex &threadLockMutex, char &key,
            std::string fileName, int numFrames, cv::Size frameSize, 
            sl::Camera &zed,
            int &numOfFiles, bool &opticalFlowDetectedFlag);
};