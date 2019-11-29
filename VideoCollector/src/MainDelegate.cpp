#include "MainDelegate.h"

MainDelegate::MainDelegate() {
    userInputKey = ' ';
    fileCount = 0;
    opticalFlowDetectedFlag = false;
    isVectorFull = false;
}

void displayGpuMat(std::mutex &threadLockMutex, cv::cuda::GpuMat &gpuMat, char &key) {
    cv::namedWindow("OriginRightView", cv::WINDOW_OPENGL);
    if (gpuMat.size().width > gpuMat.size().height*2) { // If image frame is a SideBySide case
        cv::resizeWindow("OriginRightView", gpuMat.size().width/2, gpuMat.size().height/2);
    }
    else {
        cv::resizeWindow("OriginRightView", gpuMat.size().width, gpuMat.size().height);
    }

    while(key!='q') {
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

    /*// Camera thread
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

    // Video writer thread
    std::thread videoWriter(
        &VideoWriter::writeCvGpuFramesToVideoFormat, vw, 
        std::ref(threadLockMutex), std::ref(userInputKey), 
        "/DATASETs/OpticalFlow-Motion-Dataset/", cm->getZedCameraFps(), cm->getImageFrameCvSize(), 
        std::ref(_cvLeftGpuMatFrames), std::ref(_cvRightGpuMatFrames), 
        std::ref(fileCount), std::ref(opticalFlowDetectedFlag), std::ref(isVectorFull));

    // Display trhead
    //std::thread displayFrame(displayGpuMat, std::ref(threadLockMutex), std::ref(_cvRightGpuMat), std::ref(userInputKey));        
    */
    
    std::thread getFrames(
        &CameraManager::getSideBySizeFrameFromZED, cm, 
        std::ref(threadLockMutex), 
        std::ref(_cvSideBySideGpuMat), 
        std::ref(_cvSideBySideGpuMatFrames),
        std::ref(userInputKey), std::ref(grabErrorCode),
        cm->getZedCameraFps(), std::ref(isVectorFull));

    // Display trhead
    //std::thread displayFrame(displayGpuMat, std::ref(threadLockMutex), std::ref(_cvRightGpuMat), std::ref(userInputKey));
    std::thread displayFrame(displayGpuMat, std::ref(threadLockMutex), std::ref(_cvSideBySideGpuMat), std::ref(userInputKey));

    // Optical flow thread
    fileCount ++;
    std::thread optCalc(
        &OpticalFlowManager::startSideBySideOpticalFlow, ofm, 
        std::ref(threadLockMutex), 
        std::ref(_cvSideBySideGpuMatFrames), 
        std::ref(userInputKey), std::ref(opticalFlowDetectedFlag), std::ref(grabErrorCode),
        cm->getZedCameraFps(), std::ref(isVectorFull), std::ref(fileCount));

    // Video writer thread
    std::thread videoWriter(
        &VideoWriter::writeSideBySideCvGpuFramesToVideoFormat, vw, 
        std::ref(threadLockMutex), std::ref(userInputKey), 
        "/DATASETs/OpticalFlow-Motion-Dataset/", cm->getZedCameraFps(), cm->getSideBySideImageFrameCvSize(), 
        std::ref(_cvSideBySideGpuMatFrames), 
        std::ref(fileCount), std::ref(opticalFlowDetectedFlag), std::ref(isVectorFull));

    /*std::thread videoWriter(
        &VideoWriter::writeCvGpuFramesToVideoFormat, vw, 
        std::ref(threadLockMutex), std::ref(userInputKey), 
        "/DATASETs/OpticalFlow-Motion-Dataset/", cm->getZedCameraFps(), cm->getImageFrameCvSize(), 
        std::ref(_cvSideBySideGpuMatFrames), std::ref(_cvSideBySideGpuMatFrames), 
        std::ref(fileCount), std::ref(opticalFlowDetectedFlag), std::ref(isVectorFull));*/
    
    // Keyboard interrupt thread
    std::thread interruptCall(&InterruptManager::keyInputInterrupt, im, std::ref(threadLockMutex), std::ref(userInputKey));

    getFrames.join();
    optCalc.join();
    interruptCall.join();
    displayFrame.join();
    videoWriter.join();
}