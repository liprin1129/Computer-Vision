#include <opencv2/core.hpp>
#include "CameraManager.h"
//#include <opencv2/cudaoptflow.hpp>

class OpticalFlowManager: CameraManager {
    private:
        cv::Mat _prvsLeftMat, _nextLeftMat;
        cv::Mat _prvsRightMat, _nextRightMat;

        cv::cuda::GpuMat _prvsLeftGpuMat, _nextLeftGpuMat;
        cv::cuda::GpuMat _prvsRightGpuMat, _nextRightGpuMat;
        
        cv::cuda::GpuMat _flowGpuMat;

        bool _firstGetFlag, _secondGetFlag;
        //void startOpticalFlow();
        void calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat);

    public:
        OpticalFlowManager();
        ~OpticalFlowManager();

        void startOpticalFlow();
        std::tuple<float, float> calcFlowMagnitude(const cv::cuda::GpuMat& d_flow);

        void cc(){
            for (int i=0; i<10; i++) {
                std::fprintf(stdout, "Optical!\n");
            }
        };
};