#include <iostream>

#include <thread> // std::this_thread::sleep_for
#include <chrono> // std::chrono::seconds

#include <mutex> // std::mutex for lock shared variable

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

//#include <opencv2/core.hpp>

class OpticalFlowManager {
    private:
        // Vectors containing GpuMats
        std::vector<cv::cuda::GpuMat> leftGpuMatSequence, rightGpuMatSequence;
        
        // Temporal GpuMats
        cv::cuda::GpuMat grayLeftGpuMat, grayRightGpuMat;
        cv::cuda::GpuMat resizedLeftGpuMat, resizedRightGpuMat;

        // Optical flow instace
        cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn;
        // Optical flow results
        cv::cuda::GpuMat _flowLeftGpuMat, _flowRightGpuMat;

        void calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat);

    public:
        OpticalFlowManager();
        ~OpticalFlowManager();

        //void startOpticalFlow(std::mutex &threadLockMutex, cv::cuda::GpuMat &prvsLeftGpuMat, cv::cuda::GpuMat &prvsRightGpuMat, cv::cuda::GpuMat &nextLeftGpuMat, cv::cuda::GpuMat &nextRightGpuMat, char &key, bool &opticalDetectedFlag);
        void startOpticalFlow(
            std::mutex &threadLockMutex, 
            cv::cuda::GpuMat &cvLeftGpuMat, cv::cuda::GpuMat &cvRightGpuMat, 
            char &key, bool &opticalDetectedFlag, sl::ERROR_CODE &grabErrorCode,
            int numOfAccumulatedFrames);

        std::tuple<float, float> calcFlowMeanAndStd(const cv::cuda::GpuMat& d_flow);
        std::tuple<float, float> calcFlowMinMax(const cv::cuda::GpuMat& d_flow);

        void cc(){
            for (int i=0; i<10; i++) {
                std::fprintf(stdout, "Optical!\n");
            }
        };
};