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

        // zed Mat variables
        sl::Mat _zedLeftMat, _zedRightMat;
        //sl::Mat _zedRightMat;

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

        // Open the camera
        void openCamera();
        
        void getOneFrameFromZED(std::mutex &threadLockMutex, cv::cuda::GpuMat &prvsLeftGpuMat, cv::cuda::GpuMat &prvsRightGpuMat, cv::cuda::GpuMat &nextLeftGpuMat, cv::cuda::GpuMat &nextRightGpuMat);
        void startCollectingFramesForMultiThread();

        //void displayFrames();
        void cc(){
            for (int i=0; i<10; i++) {
                std::fprintf(stdout, "Camera!\n");
            }
        };
};