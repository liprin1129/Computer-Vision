#include "CameraManager.h"

CameraManager::CameraManager() {
    initParams(); // Set configuration parameters
    openCamera(); // Open the camera
    
    // Initialise slMats
    //_zedLeftMat = sl::Mat(_zed.getResolution(), sl::MAT_TYPE_8U_C3, sl::MEM_GPU);
    //_zedRightMat = sl::Mat(_zed.getResolution(), sl::MAT_TYPE_8U_C3, sl::MEM_GPU);

    //std::this_thread::sleep_for (std::chrono::milliseconds(100));
    //displayOK = false;
    //displayFrames();
}

CameraManager::~CameraManager()
{   
    _zedLeftMat.free();
    _zedRightMat.free();
    
    _zed.close();
    std::cout << "Camera has Closed" << std::endl;
}

// Set configuration parameters
void CameraManager::initParams()
{
    _init_params.camera_resolution = sl::RESOLUTION_HD1080;
    _init_params.depth_mode = sl::DEPTH_MODE_NONE;
    _init_params.camera_disable_imu = true; // disable imu of ZED
    _runtime_parameters.enable_depth = false;
}

// Open the camera
void CameraManager::openCamera()
{
    int assertCode = 0;
    // Open the camera
    sl::ERROR_CODE err = _zed.open(_init_params);
    if (err != sl::SUCCESS) {
        std::cout << toString(err) << std::endl;
        _zed.close();
        assert(assertCode!=0); // Quit if an error occurred
    }
}

// Convert sl::Mat to cv::cuda::GpuMat for GPU
cv::cuda::GpuMat CameraManager::slMatToCvMatConverterForGPU(sl::Mat &slMat) {
    //std::cout << "GPU SL Mat" << std::endl;

    if (slMat.getMemoryType() == sl::MEM_GPU) {
        //std::cout << "GPU SL Mat" << std::endl;
        slMat.updateCPUfromGPU();
    }

    int cvType = -1;
    switch (slMat.getDataType()){
        case sl::MAT_TYPE_32F_C1: cvType = CV_32FC1; break;
        case sl::MAT_TYPE_32F_C2: cvType = CV_32FC2; break;
        case sl::MAT_TYPE_32F_C3: cvType = CV_32FC3; break;
        case sl::MAT_TYPE_32F_C4: cvType = CV_32FC4; break;
        case sl::MAT_TYPE_8U_C1: cvType = CV_8UC1; break;
        case sl::MAT_TYPE_8U_C2: cvType = CV_8UC2; break;
        case sl::MAT_TYPE_8U_C3: cvType = CV_8UC3; break;
        case sl::MAT_TYPE_8U_C4: cvType = CV_8UC4; break;
        default: break;
    }

    return cv::cuda::GpuMat(
                static_cast<int>(slMat.getHeight()), static_cast<int>(slMat.getWidth()),
                cvType,
                slMat.getPtr<sl::uchar1>(sl::MEM_GPU), slMat.getStepBytes(sl::MEM_GPU));
}

void CameraManager::getOneFrameFromZED()
{
    if(_zed.grab(_runtime_parameters) == sl::SUCCESS){
        //displayOK = true;
        // Retrieve the left image, depth image in half-resolution
        //sl::Mat _zedLeftMat(_zed.getResolution(), sl::MAT_TYPE_8U_C3, sl::MEM_GPU);
        //sl::Mat _zedRightMat(_zed.getResolution(), sl::MAT_TYPE_8U_C3, sl::MEM_GPU);
        
        _zed.retrieveImage(_zedLeftMat, sl::VIEW_LEFT, sl::MEM_GPU);
        _zed.retrieveImage(_zedRightMat, sl::VIEW_RIGHT, sl::MEM_GPU);
        
        cv::Mat cvLeftMat(slMatToCvMatConverterForGPU(_zedLeftMat));
        cv::Mat cvRightMat(slMatToCvMatConverterForGPU(_zedRightMat));

        cv::cvtColor(cvLeftMat, _cvLeftMat, cv::COLOR_BGRA2BGR);
        cv::cvtColor(cvRightMat, _cvRightMat, cv::COLOR_BGRA2BGR);

    } /*else {
        displayOK = false;
    }*/
}

/*void CameraManager::displayFrames(){
    char key = ' ';
    while (key != 'q') {
        getOneFrameFromZED();
        
        if (displayOK) {
            cv::imshow("Image", _cvRightMat);
            std::cout << _cvRightMat.channels() << std::endl;
            // Handle key event
            key = cv::waitKey(10);
        }
    }
}*/