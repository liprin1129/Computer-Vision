#include "OpticalFlowManager.h"

OpticalFlowManager::OpticalFlowManager() {
    //cm = CameraManager();

    /*
    char key = ' ';
    while (key != 'q') {
        getOneFrameFromZED();
        
        cv::Mat mat1 = cvLeftMat();

        if (!mat1.empty()) {
            cv::imshow("Image", mat1);
            //std::cout << mat1.channels() << std::endl;
            // Handle key event
            key = cv::waitKey(10);
        }
    }
    */
}

OpticalFlowManager::~OpticalFlowManager() {
    std::fprintf(stdout, "%s\n", "OpticalFlowManager destructor");
}

void OpticalFlowManager::startOpticalFlow() {
    cv::namedWindow("OriginView");
    
    char key = ' ';
    while (key != 'q') {
        getOneFrameFromZED();

        prvsRightMat = cvRightMat();

    }
}