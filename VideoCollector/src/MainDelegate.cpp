#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
    std::fprintf(stdout, "Hello World\n");
    
    //cv::namedWindow("OriginLeftView", cv::WINDOW_OPENGL);
    //cv::namedWindow("OriginRightView", cv::WINDOW_OPENGL);
    
    //CameraManager cm = CameraManager();
    //OpticalFlowManager ofm;
    std::shared_ptr<CameraManager> cm(new CameraManager);
    std::shared_ptr<OpticalFlowManager> ofm(new OpticalFlowManager);
    //std::shared_ptr<OpticalFlowManagerThreadSupported> ofmts(new OpticalFlowManagerThreadSupported);

    cm->openCamera();
    
    //ofm->startOpticalFlow();
    //std::thread cameraTest(&CameraManager::cc, cm);
    std::thread getFrames(&CameraManager::getOneFrameFromZED, cm, std::ref(threadLockMutex), std::ref(_prvsLeftGpuMat), std::ref(_prvsRightGpuMat), std::ref(_nextLeftGpuMat), std::ref(_nextRightGpuMat));
    std::thread optCalc(&OpticalFlowManager::startOpticalFlow, ofm, std::ref(threadLockMutex), std::ref(_prvsLeftGpuMat), std::ref(_prvsRightGpuMat), std::ref(_nextLeftGpuMat), std::ref(_nextRightGpuMat));
    //std::thread optThreadCalc(&OpticalFlowManagerThreadSupported::cc, ofmts);

    getFrames.join();
    optCalc.join();
    //optThreadCalc.join();
    
    //cv::destroyWindow("OriginLeftView");
    //cv::destroyWindow("OriginRightView");
}