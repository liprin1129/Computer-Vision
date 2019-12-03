#pragma once

#include "CommonHeaders.h"

#include "CameraManager.h"

class MainDelegate {
    private:
        std::mutex _threadLockMutex; // variable for mutex

        cv::cuda::GpuMat _sideBySideCvGpuMat;
        std::vector<cv::cuda::GpuMat> _cvSideBySideGpuMatFrames;

        char _userInputKey; sl::ERROR_CODE grabErrorCode;
        bool _opticalFlowDetectedFlag, _isVectorFull;

    public:
        // Constructor + Destructor
        MainDelegate();
        ~MainDelegate();

        int mainDelegation(int argc, char** argv);
};