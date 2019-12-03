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
    std::shared_ptr<InterruptManager> im(new InterruptManager);
    std::shared_ptr<OpticalFlowManager> ofm(new OpticalFlowManager);

    std::thread getFrames(  &CameraManager::getSideBySizeFrameFromZED, cm, 
                            std::ref(_threadLockMutex), 
                            std::ref(_sideBySideCvGpuMat),
                            std::ref(_cvSideBySideGpuMatFrames),
                            std::ref(_opticalFlowDetectedFlag),
                            std::ref(_userInputKey));


    std::thread optCalc (   &OpticalFlowManager::startOpticalFlow, ofm,
                            std::ref(_threadLockMutex), 
                            std::ref(_cvSideBySideGpuMatFrames),
                            cv::Size(2560, 720),
                            std::ref(_opticalFlowDetectedFlag),
                            std::ref(_userInputKey), cm->getFPS());


    std::thread displayFrame(   displayGpuMat, 
                                std::ref(_threadLockMutex), 
                                std::ref(_sideBySideCvGpuMat), 
                                std::ref(_userInputKey));


    std::thread interruptCall(&InterruptManager::keyInputInterrupt, im, std::ref(_threadLockMutex), std::ref(_userInputKey));

    getFrames.join();
    displayFrame.join();
    interruptCall.join();
    optCalc.join();

    return 0;
}