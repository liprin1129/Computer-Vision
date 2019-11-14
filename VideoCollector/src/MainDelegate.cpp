#include "MainDelegate.h"

MainDelegate::MainDelegate() {
    userInputKey = ' ';
    fileCount = 0;
    opticalFlowDetectedFlag = false;
    isVectorFull = false;
}

void displayGpuMat(std::mutex &threadLockMutex, cv::cuda::GpuMat &gpuMat, char &key) {
    cv::namedWindow("OriginRightView", cv::WINDOW_OPENGL);

    while(key!='q') {
        threadLockMutex.lock();
        
        if (!gpuMat.empty()) {
            cv::imshow("OriginRightView", gpuMat);
            //std::fprintf(stdout, "Dispaly call\n");

            //key = cv::waitKey(30);
        }
        threadLockMutex.unlock();
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }

    cv::destroyWindow("OriginRightView");
}

int MainDelegate::mainDelegation(int argc, char** argv){
    std::fprintf(stdout, "Hello World\n");
    
    std::shared_ptr<CameraManager> cm(new CameraManager);
    std::shared_ptr<OpticalFlowManager> ofm(new OpticalFlowManager);
    std::shared_ptr<InterruptManager> im(new InterruptManager);
    std::shared_ptr<VideoWriter> vw(new VideoWriter);
    std::shared_ptr<DirectoryAndFileManager> dfm(new DirectoryAndFileManager);
    
    cm->openCamera();

    // Check how many video files are in a directory and save the count to fileCount variable
    dfm->lookupDirectory("/DATASETs/OpticalFlow-Motion-Dataset/", std::ref(fileCount));

    // Camera thread
    std::thread getFrames(
        &CameraManager::getOneFrameFromZED, cm, 
        std::ref(threadLockMutex), 
        std::ref(_cvLeftGpuMat), std::ref(_cvRightGpuMat), 
        std::ref(_cvLeftGpuMatFrames), std::ref(_cvRightGpuMatFrames),
        std::ref(userInputKey), std::ref(grabErrorCode),
        cm->getZedCameraFps(), std::ref(isVectorFull));
    
    // Optical flow thread
    fileCount ++;
    std::thread optCalc(
        &OpticalFlowManager::startOpticalFlow, ofm, 
        std::ref(threadLockMutex), 
        std::ref(_cvLeftGpuMatFrames), std::ref(_cvRightGpuMatFrames), 
        std::ref(userInputKey), std::ref(opticalFlowDetectedFlag), std::ref(grabErrorCode),
        cm->getZedCameraFps(), std::ref(isVectorFull), std::ref(fileCount));
    
    // Keyboard interrupt thread
    std::thread interruptCall(&InterruptManager::keyInputInterrupt, im, std::ref(threadLockMutex), std::ref(userInputKey));
    
    // Video writer thread
    std::thread videoWriter(
        &VideoWriter::writeCvGpuFramesToVideoFormat, vw, 
        std::ref(threadLockMutex), std::ref(userInputKey), 
        "/DATASETs/OpticalFlow-Motion-Dataset/", cm->getZedCameraFps(), cm->getImageFrameCvSize(), 
        std::ref(_cvLeftGpuMatFrames), std::ref(_cvRightGpuMatFrames), 
        std::ref(fileCount), std::ref(opticalFlowDetectedFlag), std::ref(isVectorFull));

    // Display trhead
    std::thread displayFrame(displayGpuMat, std::ref(threadLockMutex), std::ref(_cvRightGpuMat), std::ref(userInputKey));
    
    getFrames.join();
    optCalc.join();
    interruptCall.join();
    displayFrame.join();
    videoWriter.join();
}