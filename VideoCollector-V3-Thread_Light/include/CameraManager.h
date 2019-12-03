#pragma once

#include "CommonHeaders.h"

#include "DirectoryAndFileManager.h"

class CameraManager {
    private:
        sl::Camera _zed;
        sl::InitParameters _init_params;
        sl::RuntimeParameters _runtime_parameters;
        int zedCameraFps; // Camera's fps
        int frameCount;

        sl::Mat _sideBySideSlMat;
        cv::cuda::GpuMat _sideBySideCvGpuMatRGBA, _sideBySideCvGpuMat;

        std::vector<cv::cuda::GpuMat> _cvSideBySideGpuMatFrames;

        // Optical flow
        bool _isOpticalFlowDetected, _isRecording;
        cv::cuda::GpuMat _cvGreyGpuMat;

        // DirectoryAndFileManager
        DirectoryAndFileManager dfm;
        int _fileCount;

        // Video Writer
        cv::VideoWriter vw;

        // PRIVATE METHODS
        void initParams();
        void openCamera();

        cv::cuda::GpuMat slMatToCvMatConverterForGPU(sl::Mat &slMat);

        void createWindow();
        void showWindow();

    protected:

    public:
        // Constructor + Destructor
        CameraManager();
        ~CameraManager();
        int getFPS(){return zedCameraFps;};
        cv::Size getFrameSize(){return cv::Size(2560, 720);};

        void getSideBySizeFrameFromZED( std::mutex &threadLockMutex,
                                        cv::cuda::GpuMat &sideBySideCvGpuMat,
                                        std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames, 
                                        bool &isOpticalFlowDetected, 
                                        char &key);

        /*void getSideBySizeFrameFromZED(
            std::mutex &threadLockMutex, 
            cv::cuda::GpuMat &sideBySideCvMat,
            std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames,
            char &key, sl::ERROR_CODE &grabErrorCode,
            bool &isVectorFull
        );*/
};