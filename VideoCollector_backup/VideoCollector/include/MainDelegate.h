#pragma once

#include <thread>

#include "CameraManager.h"
#include "OpticalFlowManager.h"
//#include "OpticalFlowManagerThreadSupported.h"
#include "InterruptManager.h"
#include "VideoWriter.h"
#include "DirectoryAndFileManager.h"

#include <mutex> // std::mutex for lock shared variable

class MainDelegate {
    private:
        std::mutex threadLockMutex; // global variable for mutex

        cv::Mat _cvLeftMat, _cvRightMat;
        cv::cuda::GpuMat _cvLeftGpuMat, _cvRightGpuMat, _cvSideBySideGpuMat;
        std::vector<cv::cuda::GpuMat> _cvLeftGpuMatFrames, _cvRightGpuMatFrames, _cvSideBySideGpuMatFrames;

        cv::cuda::GpuMat _prvsLeftGpuMat, _nextLeftGpuMat;
        cv::cuda::GpuMat _prvsRightGpuMat, _nextRightGpuMat;
        cv::cuda::GpuMat _prvsSideBySideGpuMat, _nextSideBySideGpuMat;
        cv::cuda::GpuMat _flowGpuMat;

        sl::ERROR_CODE grabErrorCode;

        char userInputKey;
        int fileCount;
        bool opticalFlowDetectedFlag, isVectorFull;

    public:
        int mainDelegation(int argc, char** argv);

        MainDelegate();
};