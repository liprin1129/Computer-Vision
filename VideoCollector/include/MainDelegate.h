#pragma once

#include <thread>

#include "CameraManager.h"
#include "OpticalFlowManager.h"
//#include "OpticalFlowManagerThreadSupported.h"
#include "InterruptManager.h"
#include "VideoWriter.h"

#include <mutex> // std::mutex for lock shared variable

class MainDelegate {
    private:
        std::mutex threadLockMutex; // global variable for mutex

        cv::cuda::GpuMat _cvLeftGpuMat, _cvRightGpuMat;

        cv::cuda::GpuMat _prvsLeftGpuMat, _nextLeftGpuMat;
        cv::cuda::GpuMat _prvsRightGpuMat, _nextRightGpuMat;
        cv::cuda::GpuMat _flowGpuMat;

        sl::ERROR_CODE grabErrorCode;

        char userInputKey;

    public:
        int mainDelegation(int argc, char** argv);

        MainDelegate();
};