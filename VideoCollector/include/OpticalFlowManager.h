#include <iostream>

#include <thread> // std::this_thread::sleep_for
#include <chrono> // std::chrono::seconds

#include <mutex> // std::mutex for lock shared variable

#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>

class OpticalFlowManager {
    private:
        cv::cuda::GpuMat _flowLeftGpuMat, _flowRightGpuMat;

        void calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat);

    public:
        OpticalFlowManager();
        ~OpticalFlowManager();

        void startOpticalFlow(std::mutex &threadLockMutex, cv::cuda::GpuMat &prvsLeftGpuMat, cv::cuda::GpuMat &prvsRightGpuMat, cv::cuda::GpuMat &nextLeftGpuMat, cv::cuda::GpuMat &nextRightGpuMat, char &key);
        
        std::tuple<float, float> calcFlowMagnitude(const cv::cuda::GpuMat& d_flow);

        void cc(){
            for (int i=0; i<10; i++) {
                std::fprintf(stdout, "Optical!\n");
            }
        };
};