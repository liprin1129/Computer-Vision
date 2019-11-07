#include "MainDelegate.h"

int MainDelegate::mainDelegation(int argc, char** argv){
    std::fprintf(stdout, "Hello World\n");
    
    //cv::namedWindow("OriginLeftView", cv::WINDOW_OPENGL);
    //cv::namedWindow("OriginRightView", cv::WINDOW_OPENGL);
    
    //CameraManager cm = CameraManager();
    //OpticalFlowManager ofm;
    std::shared_ptr<CameraManager> cm(new CameraManager);
    std::shared_ptr<OpticalFlowManager> ofm(new OpticalFlowManager);

    //ofm->startOpticalFlow();
    std::thread cameraTest(&CameraManager::cc, cm);
    std::thread optCalc(&OpticalFlowManager::cc, ofm);

    cameraTest.join();
    optCalc.join();

    //cv::destroyWindow("OriginLeftView");
    //cv::destroyWindow("OriginRightView");
}