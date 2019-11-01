#include <opencv2/core.hpp>
#include "CameraManager.h"

class OpticalFlowManager: CameraManager {
    private:
        cv::Mat prvsRightMat, nextRightMat;

    public:
        OpticalFlowManager();
        ~OpticalFlowManager();

        void startOpticalFlow();
};