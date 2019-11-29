#include "CameraManager.h"

CameraManager::CameraManager() {
    initParams(); // Set configuration parameters
    
    zedCameraFps = 60;

    // Initialise slMats
    //_zedLeftMat = sl::Mat(_zed.getResolution(), sl::MAT_TYPE_8U_C3, sl::MEM_GPU);
    //_zedRightMat = sl::Mat(_zed.getResolution(), sl::MAT_TYPE_8U_C3, sl::MEM_GPU);

    //std::this_thread::sleep_for (std::chrono::milliseconds(100));
    //displayOK = false;
    //displayFrames();
}

CameraManager::~CameraManager()
{   
    if (_zedLeftGpuMat.isInit()) {_zedLeftGpuMat.free();};
    if (_zedRightGpuMat.isInit()) {_zedRightGpuMat.free();};
    if (_zedSideBySideGpuMat.isInit()) {_zedSideBySideGpuMat.free();};
    
    _zed.close();
    std::cout << "Camera has Closed" << std::endl;
}

// Set configuration parameters
void CameraManager::initParams()
{
     _init_params.camera_resolution = sl::RESOLUTION_HD720;
    //_init_params.camera_resolution = sl::RESOLUTION_HD1080;
    //_init_params.camera_resolution = sl::RESOLUTION_HD2K;
    _init_params.depth_mode = sl::DEPTH_MODE_NONE;
    _init_params.camera_disable_imu = true; // disable imu of ZED
    _init_params.camera_fps = zedCameraFps;
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

cv::Mat slMatToCvMatConverterForCPU(sl::Mat &slMat) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (slMat.getDataType()) {
        case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(slMat.getHeight(), slMat.getWidth(), cv_type, slMat.getPtr<sl::uchar1>(sl::MEM_CPU));
}

void CameraManager::getOneFrameFromZED(
    std::mutex &threadLockMutex, 
    cv::cuda::GpuMat &cvLeftGpuMat, cv::cuda::GpuMat &cvRightGpuMat, 
    std::vector<cv::cuda::GpuMat> &cvLeftGpuMatFrames, std::vector<cv::cuda::GpuMat> &cvRightGpuMatFrames, 
    char &key, sl::ERROR_CODE &grabErrorCode,
    int numFrames, bool &isVectorFull) 
    {   
        std::vector<cv::cuda::GpuMat> leftFrames, rightFrames;

        while (key != 'q') {
            const int64 start = cv::getTickCount();
            
            if (_zed.grab(_runtime_parameters) == sl::SUCCESS){

                //std::cout << "Camera Thread: " << key << std::endl << std::flush;

                _zed.retrieveImage(_zedLeftGpuMat, sl::VIEW_LEFT, sl::MEM_GPU);
                _zed.retrieveImage(_zedRightGpuMat, sl::VIEW_RIGHT, sl::MEM_GPU);

                cv::cuda::cvtColor(slMatToCvMatConverterForGPU(_zedLeftGpuMat), _cvLeftGpuMat, cv::COLOR_BGRA2BGR);
                cv::cuda::cvtColor(slMatToCvMatConverterForGPU(_zedRightGpuMat), _cvRightGpuMat, cv::COLOR_BGRA2BGR);

                // Lock
                // threadLockMutex.lock();
                
                cvLeftGpuMat = _cvLeftGpuMat.clone();
                cvRightGpuMat = _cvRightGpuMat.clone();

                // Unlock
                // threadLockMutex.unlock();
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));
                
                grabErrorCode = sl::SUCCESS;

                // Save frame to match fps
                leftFrames.push_back(_cvLeftGpuMat.clone());
                rightFrames.push_back(_cvRightGpuMat.clone());

                //std::cout << "[Camera] cvLeftGpuMat refcount: " << *_cvLeftGpuMat.refcount << "\t vector size: " << leftFrames.size() << " : " << rightFrames.size() << std::endl;

                threadLockMutex.lock();

                if (leftFrames.size() >= numFrames or rightFrames.size() >= numFrames) {
                    // Lock
                    //threadLockMutex.lock();
                    // std::cout << "\n[Camera] Vector full!\n";

                    cvLeftGpuMatFrames.assign(leftFrames.begin(), leftFrames.end());
                    cvRightGpuMatFrames.assign(rightFrames.begin(), rightFrames.end());

                    //std::cout << "[Camera] vector size: " << cvLeftGpuMatFrames.size() << " : " << cvRightGpuMatFrames.size() << std::endl;

                    isVectorFull = true;
                    
                    // Unlock
                    //threadLockMutex.unlock();
                    // std::this_thread::sleep_for(std::chrono::milliseconds(1));

                    // Clear vectors
                    leftFrames.clear(); rightFrames.clear();
                } else {
                    // threadLockMutex.lock();

                    isVectorFull = false;

                    // threadLockMutex.unlock();
                    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                threadLockMutex.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            else {
                grabErrorCode = sl::ERROR_CODE_FAILURE;
            }

            const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
            //std::fprintf(stdout, "CameraManager time : %lf sec\n", timeSec);
        }

        //_zed.disableRecording();
}

void CameraManager::getSideBySizeFrameFromZED(
    std::mutex &threadLockMutex, 
    cv::cuda::GpuMat &cvSideBySideGpuMat,
    std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames,
    char &key, sl::ERROR_CODE &grabErrorCode,
    int numFrames, bool &isVectorFull) 
    {   
        std::vector<cv::cuda::GpuMat> sideBySideFrames;

        while (key != 'q') {
            //const int64 start = cv::getTickCount();
            
            if (_zed.grab(_runtime_parameters) == sl::SUCCESS){
                _zed.record();
                //std::cout << "Camera Thread: " << key << std::endl << std::flush;
                _zed.retrieveImage(_zedSideBySideGpuMat, sl::VIEW_SIDE_BY_SIDE, sl::MEM_GPU);

                cv::cuda::cvtColor(slMatToCvMatConverterForGPU(_zedSideBySideGpuMat), cvSideBySideGpuMat, cv::COLOR_BGRA2BGR);

                // Lock
                // threadLockMutex.lock();
                
                //cvSideBySideGpuMat = _cvSideBySideGpuMat.clone();

                // Unlock
                // threadLockMutex.unlock();
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));
                
                grabErrorCode = sl::SUCCESS;

                // Save frame to match fps
                sideBySideFrames.push_back(cvSideBySideGpuMat.clone());
                //std::cout << "[Camera] _cvSideBySideGpuMat refcount: " << *cvSideBySideGpuMat.refcount << "\t vector size: " << cvSideBySideGpuMat.size() << std::endl;

                threadLockMutex.lock();

                if (sideBySideFrames.size() >= numFrames) {
                    // Lock
                    //threadLockMutex.lock();
                    // std::cout << "\n[Camera] Vector full!\n";

                    cvSideBySideGpuMatFrames.assign(sideBySideFrames.begin(), sideBySideFrames.end());

                    //std::cout << "[Camera] vector size: " << cvLeftGpuMatFrames.size() << " : " << cvRightGpuMatFrames.size() << std::endl;

                    isVectorFull = true;
                    
                    // Unlock
                    //threadLockMutex.unlock();
                    // std::this_thread::sleep_for(std::chrono::milliseconds(1));

                    // Clear vectors
                    sideBySideFrames.clear();
                } else {
                    // threadLockMutex.lock();

                    isVectorFull = false;

                    // threadLockMutex.unlock();
                    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                threadLockMutex.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                //std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            else {
                grabErrorCode = sl::ERROR_CODE_FAILURE;
            }

            //const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
            //std::fprintf(stdout, "CameraManager time : %lf sec\n", timeSec);
        }

        //_zed.disableRecording();
}

void CameraManager::startCollectingFramesForMultiThread() {
    openCamera(); // Open the camera

    /* 
    while (true) {
        getOneFrameFromZED();
    }
    */
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