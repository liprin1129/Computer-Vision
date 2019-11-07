#pragma once

#include <thread>

#include "CameraManager.h"
#include "OpticalFlowManager.h"
//#include "OpticalFlowManagerThreadSupported.h"

#include <mutex> // std::mutex for lock shared variable
static std::mutex threadLockMutex; // global variable for mutex

class MainDelegate {
    private:
        cv::cuda::GpuMat _cvLeftGpuMat, _cvRightGpuMat;

        cv::cuda::GpuMat _prvsLeftGpuMat, _nextLeftGpuMat;
        cv::cuda::GpuMat _prvsRightGpuMat, _nextRightGpuMat;
        cv::cuda::GpuMat _flowGpuMat;

        sl::ERROR_CODE grabErrorCode;
    public:
        int mainDelegation(int argc, char** argv);
};