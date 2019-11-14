#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include <thread>
#include <chrono>
#include <mutex>

class VideoWriter {
    private:
        cv::VideoWriter vw;
        cv::Mat originLeftCpuMat;
        std::vector<cv::cuda::GpuMat> originLeftGpuMat, originRightGpuMat;

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

        void writeSlFramesToVideoFormat(
            std::mutex &threadLockMutex, char &key,
            std::string fileName, int numFrames, cv::Size frameSize, 
            sl::Camera &zed,
            int &numOfFiles, bool &opticalFlowDetectedFlag);
};