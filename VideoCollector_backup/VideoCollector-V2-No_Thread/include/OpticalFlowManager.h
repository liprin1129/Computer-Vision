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
        //std::vector<cv::cuda::GpuMat> leftGpuMatSequence, rightGpuMatSequence;
        
        // Temporal GpuMats
        cv::cuda::GpuMat _greyPrvsLeftGpuMat, _greyPrvsRightGpuMat, _greyPrvsSideBySideGpuMat;
        cv::cuda::GpuMat _greyNextLeftGpuMat, _greyNextRightGpuMat, _greyNextSideBySideGpuMat;
        //cv::cuda::GpuMat _resizedLeftGpuMat, _resizedRightGpuMat, _resizedSideBySideGpuMat;
        std::vector<cv::cuda::GpuMat> _originLeftGpuMat, _originRightGpuMat, _originSideBySideGpuMat;
        
        std::vector<cv::cuda::GpuMat> _cvSideBySideGpuMatFrames;

        // Optical flow instace
        cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn;
        // Optical flow results
        cv::cuda::GpuMat _flowLeftGpuMat, _flowRightGpuMat, _flowSideBySideGpuMat;

        void calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat);

    public:
        OpticalFlowManager();
        ~OpticalFlowManager();

        int startOpticalFlow(std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFramesThread);
        
        //void startOpticalFlow(std::mutex &threadLockMutex, cv::cuda::GpuMat &prvsLeftGpuMat, cv::cuda::GpuMat &prvsRightGpuMat, cv::cuda::GpuMat &nextLeftGpuMat, cv::cuda::GpuMat &nextRightGpuMat, char &key, bool &opticalDetectedFlag);
        void startOpticalFlowThread(
            std::mutex &threadLockMutex, 
            //cv::cuda::GpuMat &cvLeftGpuMat, cv::cuda::GpuMat &cvRightGpuMat, 
            std::vector<cv::cuda::GpuMat> &cvLeftGpuMatFrames, std::vector<cv::cuda::GpuMat> &cvRightGpuMatFrames,
            char &key, bool &opticalDetectedFlag, sl::ERROR_CODE &grabErrorCode,
            int numOfAccumulatedFrames, bool &isVectorFull, int &fileCount);

        void startSideBySideOpticalFlowThread(
            std::mutex &threadLockMutex, 
            //cv::cuda::GpuMat &cvLeftGpuMat, cv::cuda::GpuMat &cvRightGpuMat, 
            std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames,
            char &key, bool &opticalDetectedFlag, sl::ERROR_CODE &grabErrorCode,
            int numOfAccumulatedFrames, bool &isVectorFull, int &fileCount);

        std::tuple<float, float> calcFlowMeanAndStd(const cv::cuda::GpuMat& d_flow);
        std::tuple<float, float> calcFlowMinMax(const cv::cuda::GpuMat& d_flow);

        void cc(){
            for (int i=0; i<10; i++) {
                std::fprintf(stdout, "Optical!\n");
            }
        };
};