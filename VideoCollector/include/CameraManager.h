#pragma once

#include <iostream>
#include <sl/Camera.hpp>
#include <sl/Core.hpp>

#include <opencv2/opencv.hpp>

#include <cassert> // assert Macro
#include <thread> // std::this_thread::sleep_for
#include <chrono> // std::chrono::seconds

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

        // Open the camera
        void openCamera();

        // Convert sl::Mat to cv::Mat for GPU
        cv::cuda::GpuMat slMatToCvMatConverterForGPU(sl::Mat &slMat);
        
    protected:
        //sl::ERROR_CODE _grabErrorCode;
        //cv::Mat _cvLeftMat, _cvRightMat;
        cv::cuda::GpuMat _cvLeftGpuMat, _cvRightGpuMat;
    public:
        CameraManager();
        ~CameraManager();

        // Getters
        //cv::Mat cvLeftMat() {return _cvLeftMat;};
        //cv::Mat cvRightMat() {return _cvRightMat;};

        sl::ERROR_CODE getOneFrameFromZED();
        void displayFrames();
};