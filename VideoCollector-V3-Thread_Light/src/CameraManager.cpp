#include "CameraManager.h"

CameraManager::CameraManager() {
    initParams(); // Set configuration parameters
    openCamera();

    // createWindow();
    // zedCameraFps = _zed.getCameraFPS(); // Get FPS
    zedCameraFps = 60; // Get FPS

    frameCount = 0;

    //ofm = OpticalFlowManager();
    _isOpticalFlowDetected = false;
    _isRecording = false;
}

CameraManager::~CameraManager() {

    _sideBySideSlMat.free();

    // cv::destroyWindow("Window");

    _zed.close();

    std::fprintf(stdout, "CameraManager destructor\n");
}


/**
 * @brief INITIALIZE PARAMETERS
 * 
 */
void CameraManager::initParams() {    
    _init_params.camera_resolution = sl::RESOLUTION_HD720;
    // _init_params.camera_resolution = sl::RESOLUTION_HD1080;
    //_init_params.camera_resolution = sl::RESOLUTION_HD2K;
    _init_params.depth_mode = sl::DEPTH_MODE_NONE;
    _init_params.camera_disable_imu = true; // disable imu of ZED
    _init_params.camera_fps = zedCameraFps;
    _runtime_parameters.enable_depth = false;
}

/**
 * @brief OPEN THE CAMERA
 * 
 */
void CameraManager::openCamera() {
    int assertCode = 0;
    // Open the camera
    sl::ERROR_CODE err = _zed.open(_init_params);

    if (err != sl::SUCCESS) {
        std::cout << toString(err) << std::endl;
        _zed.close();
        assert(assertCode!=0); // Quit if an error occurred
    }
}

/**
 * @brief Creat a CV OpenGL window
 * 
 */
void CameraManager::createWindow() {
    cv::namedWindow("Window", cv::WINDOW_OPENGL);

    sl::Resolution resolution = _zed.getResolution();

    cv::resizeWindow("Window", resolution.width, resolution.height/2);
}

/**
 * @brief Show CV GPU Mat
 * 
 */
void CameraManager::showWindow() {
    if (!_sideBySideCvGpuMat.empty()) {
        cv::imshow("Window", _sideBySideCvGpuMat);
    }
}

cv::cuda::GpuMat CameraManager::slMatToCvMatConverterForGPU(sl::Mat &slMat) {
    // CONVERT sl::Mat TO cv::cuda::GpuMat FOR GPU

    /*if (slMat.getMemoryType() == sl::MEM_GPU) {
        slMat.updateCPUfromGPU();
    }*/

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

    return cv::cuda::GpuMat(slMat.getHeight(), slMat.getWidth(), cvType, slMat.getPtr<sl::uchar1>(sl::MEM_GPU));
    /*return cv::cuda::GpuMat(
                static_cast<int>(slMat.getHeight()), static_cast<int>(slMat.getWidth()),
                cvType,
                slMat.getPtr<sl::uchar1>(sl::MEM_GPU), slMat.getStepBytes(sl::MEM_GPU));*/
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(sl::Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cvType = -1;
    switch (input.getDataType()) {
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

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cvType, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}


void CameraManager::getSideBySizeFrameFromZED(  std::mutex &threadLockMutex,
                                                cv::cuda::GpuMat &sideBySideCvGpuMat,
                                                std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames, 
                                                bool &isOpticalFlowDetected,
                                                char &key) {
    dfm.lookupDirectory("/DATASETs/OpticalFlow-Motion-Dataset/", _fileCount);
    int currentCount = _fileCount;

    // char key = ' ';
    while (key != 'q') {
        // std::fprintf(stdout, "Camera\n");

        if (_zed.grab(_runtime_parameters) == sl::SUCCESS) {
            _zed.retrieveImage(_sideBySideSlMat, sl::VIEW_SIDE_BY_SIDE, sl::MEM_GPU);
            _sideBySideCvGpuMatRGBA = slMatToCvMatConverterForGPU(_sideBySideSlMat);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            cv::cuda::cvtColor(_sideBySideCvGpuMatRGBA, _sideBySideCvGpuMat, cv::COLOR_RGBA2RGB);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            // cv::cuda::cvtColor(slMatToCvMatConverterForGPU(_sideBySideSlMat), _sideBySideCvGpuMat, cv::COLOR_RGBA2GRAY);

            /*
            if (isOpticalFlowDetected == true) {
                if (_fileCount==currentCount) {
                    std::string saveFile = "/DATASETs/OpticalFlow-Motion-Dataset/" + std::to_string(_fileCount) + ".avi";
                    sl::String pathOutput(saveFile.c_str());
                    // _zed.enableRecording(pathOutput, sl::SVO_COMPRESSION_MODE_LOSSLESS);
                    vw = cv::VideoWriter(saveFile, CV_FOURCC('H','2','6','4'), zedCameraFps/2, _sideBySideCvGpuMat.size());
                    ++currentCount;
                }

                cv::Mat mat(_sideBySideCvGpuMat);
                vw.write(mat);
               _isRecording = true;
                // std::fprintf(stdout, "Recording: %d\n", _fileCount);
            } 
            else if (_isOpticalFlowDetected == false and _isRecording == true) {
                ++_fileCount;
                //  std::fprintf(stdout, "Stop recording: %d\n", _fileCount);
                _isRecording = false;
                
                // _zed.disableRecording();
            }

            */
            // std::fprintf(stdout, "Frame: %d", frameCount);

            _cvSideBySideGpuMatFrames.push_back(_sideBySideCvGpuMat.clone());
            
            threadLockMutex.lock();
            cvSideBySideGpuMatFrames.assign(_cvSideBySideGpuMatFrames.begin(), _cvSideBySideGpuMatFrames.end()); // For OpticalFlow
            sideBySideCvGpuMat = _sideBySideCvGpuMat.clone(); // FOR DISPLAY
            // std::cout << "Camera: " << cv::cuda::sum(_cvSideBySideGpuMatFrames[0]) << " | " << cv::cuda::sum(_cvSideBySideGpuMatFrames[1]) << std::endl;
            // _isOpticalFlowDetected = isOpticalFlowDetected;
            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
            // std::fprintf(stdout, "Camera: %d\n", static_cast<int>(_cvSideBySideGpuMatFrames.size()));

            ++frameCount;
            if (frameCount == zedCameraFps) {
                _cvSideBySideGpuMatFrames.clear();
                //_isOpticalFlowDetected = ofm.startOpticalFlow(_cvSideBySideGpuMatFrames);

                frameCount = 0;
            };

            // showWindow();
        }

        // key = cv::waitKey(1);
    }

}

/*void CameraManager::getSideBySizeFrameFromZED( std::mutex &threadLockMutex, 
                                cv::cuda::GpuMat &sideBySideCvMat,
                                std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames,
                                char &key, sl::ERROR_CODE &grabErrorCode,
                                bool &isVectorFull) {
    while (key != 'q') {
        grabErrorCode = _zed.grab(_runtime_parameters);

        if (grabErrorCode == sl::SUCCESS) {
            //std::fprintf(stdout, "Capture\n");
            _zed.retrieveImage(_sideBySideSlMat, sl::VIEW_SIDE_BY_SIDE, sl::MEM_GPU);
            _sideBySideCvGpuMat = slMatToCvMatConverterForGPU(_sideBySideSlMat);

            threadLockMutex.lock();
            //_sideBySideCvGpuMat = sideBySideCvMat.clone();
            //showWindow();
            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}*/