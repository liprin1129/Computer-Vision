#include "MainDelegate.h"

MainDelegate::MainDelegate() {
    userInputKey = ' ';
}

int MainDelegate::mainDelegation(int argc, char** argv){
    std::fprintf(stdout, "Hello World\n");
    
    //cv::namedWindow("OriginLeftView", cv::WINDOW_OPENGL);
    //cv::namedWindow("OriginRightView", cv::WINDOW_OPENGL);
    
    std::shared_ptr<CameraManager> cm(new CameraManager);
    std::shared_ptr<OpticalFlowManager> ofm(new OpticalFlowManager);
    std::shared_ptr<InterruptManager> im(new InterruptManager);
    std::shared_ptr<VideoWriter> vw(new VideoWriter);

    cm->openCamera();

    // Camera thread    
    std::thread getFrames(&CameraManager::getOneFrameFromZED, cm, std::ref(threadLockMutex), std::ref(_prvsLeftGpuMat), std::ref(_prvsRightGpuMat), std::ref(_nextLeftGpuMat), std::ref(_nextRightGpuMat), std::ref(userInputKey));
    // Optical flow thread
    std::thread optCalc(&OpticalFlowManager::startOpticalFlow, ofm, std::ref(threadLockMutex), std::ref(_prvsLeftGpuMat), std::ref(_prvsRightGpuMat), std::ref(_nextLeftGpuMat), std::ref(_nextRightGpuMat), std::ref(userInputKey));
    // Keyboard interrupt thread
    std::thread interruptCall(&InterruptManager::keyInputInterrupt, im, std::ref(threadLockMutex), std::ref(userInputKey));
    // Video writer thread
    //cv::Size imgSize = cm->getImageFrameCvSize();
    std::thread videoWriter(&VideoWriter::writeFramesToVideoFormat, vw, std::ref(threadLockMutex), std::ref(userInputKey), "./output.avi", 10, cm->getImageFrameCvSize(), std::ref(_prvsLeftGpuMat));

    getFrames.join();
    optCalc.join();
    interruptCall.join();
    videoWriter.join();
    
    //userInput.join();
    //optThreadCalc.join();
    
    //cv::destroyWindow("OriginLeftView");
    //cv::destroyWindow("OriginRightView");
}