#pragma once

#include <iostream>
#include <sl/Camera.hpp>
#include <sl/Core.hpp>

#include <opencv2/opencv.hpp>

//#include <cassert> // assert Macro

#include <thread> // std::this_thread::sleep_for
#include <chrono> // std::chrono::seconds

#include <mutex> // std::mutex for lock shared variable


//static std::mutex threadLockMutex; // global variable for mutex

class CameraManager {
    private:
        sl::Camera _zed;

        sl::InitParameters _init_params;
        sl::RuntimeParameters _runtime_parameters;
        sl::Resolution _image_size;

        int zedCameraFps;

        // zed Mat variables
        sl::Mat _zedLeftGpuMat, _zedRightGpuMat, _zedSideBySideGpuMat;

        cv::cuda::GpuMat _cvLeftGpuMat, _cvRightGpuMat, _cvSideBySideGpuMatRGBA, _cvSideBySideGpuMat;

        //bool displayOK;
        
        // Set configuration parameters
        void initParams();
        
        // Convert sl::Mat to cv::Mat for GPU
        cv::cuda::GpuMat slMatToCvMatConverterForGPU(sl::Mat &slMat);

        
    protected:
        //sl::ERROR_CODE _grabErrorCode;
        //cv::Mat _cvLeftMat, _cvRightMat;
        //cv::cuda::GpuMat _cvLeftGpuMat, _cvRightGpuMat;
    
    public:

        CameraManager();
        ~CameraManager();

        // Getters
        /* cv::cuda::GpuMat cvLeftGpuMat() {
            std::lock_guard<std::mutex> gaurd(threadLockMutex);
            return _cvLeftGpuMat;
        };

        cv::cuda::GpuMat cvRightGpuMat() {
            std::lock_guard<std::mutex> gaurd(threadLockMutex);
            return _cvRightGpuMat;
        }; */
        cv::Size getImageFrameCvSize() {
            auto res = _zed.getResolution();
            return cv::Size(res.width, res.height);
        }

        cv::Size getSideBySideImageFrameCvSize() {
            auto res = _zed.getResolution();
            return cv::Size(res.width*2, res.height);
        }

        int getZedCameraFps() {return zedCameraFps;};
        sl::Camera &getZed() {return _zed;};

        // Open the camera
        void openCamera();
        
        void getOneFrameFromZED(
            std::mutex &threadLockMutex, 
            cv::cuda::GpuMat &cvLeftGpuMat, cv::cuda::GpuMat &cvRightGpuMat, 
            std::vector<cv::cuda::GpuMat> &cvLeftGpuMatFrames, std::vector<cv::cuda::GpuMat> &cvRightGpuMatFrames, 
            char &key, sl::ERROR_CODE &grabErrorCode,
            int numFrames, bool &isVectorFull);

        void getSideBySizeFrameFromZED(
            std::mutex &threadLockMutex, 
            cv::cuda::GpuMat &cvSideBySideGpuMat, 
            std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames,
            char &key, sl::ERROR_CODE &grabErrorCode,
            int numFrames, bool &isVectorFull);

        void startCollectingFramesForMultiThread();

        //void displayFrames();
        void cc(){
            for (int i=0; i<10; i++) {
                std::fprintf(stdout, "Camera!\n");
            }
        };
};