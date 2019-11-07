#include "OpticalFlowManager.h"

OpticalFlowManager::OpticalFlowManager() {
    std::fprintf(stdout, "%s\n", "OpticalFlowManager constructor");
}

OpticalFlowManager::~OpticalFlowManager() {
    std::fprintf(stdout, "%s\n", "OpticalFlowManager destructor");
}

std::tuple<float, float> OpticalFlowManager::calcFlowMagnitude(const cv::cuda::GpuMat& d_flow) {
    cv::cuda::GpuMat xyPlanes[2];
    cv::cuda::split(d_flow, xyPlanes);

    cv::cuda::GpuMat sqrGpuX, sqrGpuY, addSqrGpuXY, sqrtGpuXY;

    cv::cuda::sqr(xyPlanes[0], sqrGpuX); 
    cv::cuda::sqr(xyPlanes[1], sqrGpuY);
    cv::cuda::add(sqrGpuX, sqrGpuY, addSqrGpuXY);
    cv::cuda::sqrt(addSqrGpuXY, sqrtGpuXY);
    
    //auto a = cv::Mat(sqrtGpuXY);
    //std::cout << "GPU: " << a.size() << std::endl;
    //std::cout << "GPU: " << a.channels() << std::endl;

    cv::Mat sqrtXY, meanVal, stdVal;
    sqrtGpuXY.download(sqrtXY);
    cv::meanStdDev(sqrtXY, meanVal, stdVal);

    return std::make_tuple(
        static_cast<float>(meanVal.at<double>(0)), 
        static_cast<float>(stdVal.at<double>(0)));
}

void OpticalFlowManager::startOpticalFlow(std::mutex &threadLockMutex, cv::cuda::GpuMat &prvsLeftGpuMat, cv::cuda::GpuMat &prvsRightGpuMat, cv::cuda::GpuMat &nextLeftGpuMat, cv::cuda::GpuMat &nextRightGpuMat) {
    cv::cuda::GpuMat grayPrvsLeftGpuMat, grayPrvsRightGpuMat;
    cv::cuda::GpuMat grayNextLeftGpuMat, grayNextRightGpuMat;

    cv::cuda::GpuMat resizedPrvsLeftGpuMat, resizedPrvsRightGpuMat;
    cv::cuda::GpuMat resizedNextLeftGpuMat, resizedNextRightGpuMat;

    while(true) {
        threadLockMutex.lock();
        //std::cout << "\tOptical Thread." << std::endl << std::flush;
        
        std::cout << prvsLeftGpuMat.empty() << prvsRightGpuMat.empty() << nextLeftGpuMat.empty() << nextRightGpuMat.empty() << std::endl;

        if (!prvsLeftGpuMat.empty() and !prvsRightGpuMat.empty() and !nextLeftGpuMat.empty() and !nextRightGpuMat.empty()) {
            //std::cout << prvsLeftGpuMat.size() << std::endl;

            // Convert to Gray scale
            cv::cuda::cvtColor(prvsLeftGpuMat, grayPrvsLeftGpuMat, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(prvsRightGpuMat, grayPrvsRightGpuMat, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(nextLeftGpuMat, grayNextLeftGpuMat, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(nextRightGpuMat, grayNextRightGpuMat, cv::COLOR_BGR2GRAY);

            cv::cuda::resize(grayPrvsLeftGpuMat, resizedPrvsLeftGpuMat, cv::Size(480, 270));
            cv::cuda::resize(grayPrvsRightGpuMat, resizedPrvsRightGpuMat, cv::Size(480, 270));

            cv::cuda::resize(grayNextLeftGpuMat, resizedNextLeftGpuMat, cv::Size(480, 270));
            cv::cuda::resize(grayNextRightGpuMat, resizedNextRightGpuMat, cv::Size(480, 270));

            const int64 start = cv::getTickCount();

            calcOpticalFlowGPU(resizedPrvsLeftGpuMat, resizedNextLeftGpuMat, _flowLeftGpuMat);
            calcOpticalFlowGPU(resizedPrvsRightGpuMat, resizedNextRightGpuMat, _flowRightGpuMat);
            
            auto [lMean, lStd] = calcFlowMagnitude(_flowLeftGpuMat);
            auto [rMean, rStd] = calcFlowMagnitude(_flowRightGpuMat);
            
            const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();

            std::fprintf(stdout, "FlowCalc time : %lf sec\t", timeSec);
            //std::cout << "FlowCalc time: "<< timeSec << "sec\n" << std::flush;
            std::fprintf(stdout, "Maen: %f, Std: %f\n", lMean, lStd);
        }
        
        
        threadLockMutex.unlock();
        //std::this_thread::sleep_for(std::chrono::seconds(1));
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void OpticalFlowManager::calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat) {
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
    farn->calc(prvsGpuMat, nextGpuMat, xyVelocityGpuMat);
}