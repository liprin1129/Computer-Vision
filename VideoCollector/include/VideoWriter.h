#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <mutex>

class VideoWriter {
    private:
        cv::VideoWriter vw;

    public:
        VideoWriter();

        void writeFramesToVideoFormat(
            std::mutex &threadLockMutex, char &key,
            std::string fileName, int numFrames, cv::Size frameSize, 
            cv::cuda::GpuMat &originGpuMat);
};