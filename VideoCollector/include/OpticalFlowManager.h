#include <opencv2/core.hpp>
#include "CameraManager.h"
//#include <opencv2/cudaoptflow.hpp>

class OpticalFlowManager: CameraManager {
    private:
        cv::Mat prvsLeftMat, nextLeftMat;
        cv::Mat prvsRightMat, nextRightMat;

        cv::cuda::GpuMat prvsLeftGpuMat, nextLeftGpuMat;
        cv::cuda::GpuMat prvsRightGpuMat, nextRightGpuMat;
        
        cv::cuda::GpuMat flowGpuMat;

        bool firstGetFlag, secondGetFlag;

    public:
        OpticalFlowManager();
        ~OpticalFlowManager();

        void startOpticalFlow();
};