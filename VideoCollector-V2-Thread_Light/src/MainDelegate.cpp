#include "MainDelegate.h"

/**
 * @brief Construct a new Main Delegate:: Main Delegate object
 * 
 */
MainDelegate::MainDelegate() {
    std::fprintf(stdout, "MainDelegate constructor\n");

    _opticalFlowDetectedFlag = false;
}


/**
 * @brief Construct a new Main Delegate:: Main Delegate object
 * 
 */
MainDelegate::~MainDelegate() {
    _sideBySideCvGpuMat.release();
    std::fprintf(stdout, "MainDelegate destructor\n");
}

/**
 * @brief Show CV GPU Mat
 * 
 * @param threadLockMutex 
 * @param gpuMat 
 * @param key 
 */
void displayGpuMat(std::mutex &threadLockMutex, cv::cuda::GpuMat &gpuMat, char &key) {
    cv::namedWindow("OriginRightView", cv::WINDOW_OPENGL);

    if (gpuMat.size().width > gpuMat.size().height*2) { // If image frame is a SideBySide case
        cv::resizeWindow("OriginRightView", gpuMat.size().width/2, gpuMat.size().height/2);
    }
    else {
        cv::resizeWindow("OriginRightView", gpuMat.size().width, gpuMat.size().height);
    }

    while(key!='q') {
        //std::fprintf(stdout, "Display\n");
        threadLockMutex.lock();
        
        //std::cout << gpuMat.size() << std::endl;
        
        if (!gpuMat.empty()) {
            cv::imshow("OriginRightView", gpuMat);
            //std::fprintf(stdout, "Dispaly call\n");
            //key = cv::waitKey(30);
        }

        threadLockMutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    cv::destroyWindow("OriginRightView");
}


/**
 * @brief Main function delegation
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int MainDelegate::mainDelegation(int argc, char** argv){
    std::fprintf(stdout, "Hello World\n");

    std::shared_ptr<CameraManager> cm(new CameraManager);
    cm->getSideBySizeFrameFromZED();

    return 0;
}