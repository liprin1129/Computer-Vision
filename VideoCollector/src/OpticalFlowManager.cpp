#include "OpticalFlowManager.h"

OpticalFlowManager::OpticalFlowManager() {
    farn = cv::cuda::FarnebackOpticalFlow::create();
    std::fprintf(stdout, "%s\n", "OpticalFlowManager constructor");
}

OpticalFlowManager::~OpticalFlowManager() {
    std::fprintf(stdout, "%s\n", "OpticalFlowManager destructor");
}

std::tuple<float, float> OpticalFlowManager::calcFlowMeanAndStd(const cv::cuda::GpuMat& d_flow) {
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

std::tuple<float, float> OpticalFlowManager::calcFlowMinMax(const cv::cuda::GpuMat& d_flow) {
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

    double min_loc, max_loc;
    cv::minMaxLoc(sqrtXY, &min_loc, &max_loc);
    //std::fprintf(stdout, "Max: %lf, Min: %lf\n", max_loc, min_loc);

    return std::make_tuple(
        static_cast<float>(min_loc), 
        static_cast<float>(max_loc));
}

/*
void OpticalFlowManager::startOpticalFlow(std::mutex &threadLockMutex, cv::cuda::GpuMat &prvsLeftGpuMat, cv::cuda::GpuMat &prvsRightGpuMat, cv::cuda::GpuMat &nextLeftGpuMat, cv::cuda::GpuMat &nextRightGpuMat, char &key, bool &opticalDetectedFlag) {
    cv::cuda::GpuMat grayPrvsLeftGpuMat, grayPrvsRightGpuMat;
    cv::cuda::GpuMat grayNextLeftGpuMat, grayNextRightGpuMat;

    cv::cuda::GpuMat resizedPrvsLeftGpuMat, resizedPrvsRightGpuMat;
    cv::cuda::GpuMat resizedNextLeftGpuMat, resizedNextRightGpuMat;

    while(key != 'q') {
        threadLockMutex.lock();
        //std::cout << "\tOptical Thread." << std::endl << std::flush;
        //std::cout << "\tOptical Thread: " << key << std::endl << std::flush;
        //std::cout << prvsLeftGpuMat.empty() << prvsRightGpuMat.empty() << nextLeftGpuMat.empty() << nextRightGpuMat.empty() << std::endl;

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
            
            // Calculate the magintude of optical flolw
            auto [lMean, lStd] = calcFlowMagnitude(_flowLeftGpuMat);
            auto [rMean, rStd] = calcFlowMagnitude(_flowRightGpuMat);

            const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();

            std::fprintf(stdout, "FlowCalc time : %lf sec\t", timeSec);
            std::fprintf(stdout, "lMaen: %f, Std: %f, rMaen: %f, Std: %f\n", lMean, lStd, rMean, rStd);
        }
        
        threadLockMutex.unlock();
        //std::this_thread::sleep_for(std::chrono::seconds(1));
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}
*/

void OpticalFlowManager::startOpticalFlow(
    std::mutex &threadLockMutex, 
    //cv::cuda::GpuMat &cvLeftGpuMat, cv::cuda::GpuMat &cvRightGpuMat, 
    std::vector<cv::cuda::GpuMat> &cvLeftGpuMatFrames, std::vector<cv::cuda::GpuMat> &cvRightGpuMatFrames,
    char &key, bool &opticalDetectedFlag, sl::ERROR_CODE &grabErrorCode,
    int numOfAccumulatedFrames, bool &isVectorFull, int &fileCount) {

    //std::vector<cv::cuda::GpuMat> leftGpuMatSequence(numOfAccumulatedFrames), rightGpuMatSequence(numOfAccumulatedFrames);
    int vectorCounter = 0;
    int boolCounter = 0;
    bool recording = false;

    //cv::cuda::GpuMat leftGpuMatSequence[numOfAccumulatedFrames], rightGpuMatSequence[numOfAccumulatedFrames];

    while (key!='q') {
        const int64 start = cv::getTickCount();

        threadLockMutex.lock();
        //std::cout << "\t[Optic]\n";
        
        if (isVectorFull == true) {
            // std::cout << "\t[Optic] FULL\n";
            originLeftGpuMat.assign(cvLeftGpuMatFrames.begin(), cvLeftGpuMatFrames.end());
            originRightGpuMat.assign(cvRightGpuMatFrames.begin(), cvRightGpuMatFrames.end());

            //cvLeftGpuMatFrames.clear();
            //cvRightGpuMatFrames.clear();

            //isVectorFull = false;
            //key = 'q';
        }

        //std::cout << "\t[Optic ] vector size: " << originLeftGpuMat.size() << " : " << originRightGpuMat.size() << std::endl;
        
        threadLockMutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        if (originLeftGpuMat.size() >= numOfAccumulatedFrames or originRightGpuMat.size() >= numOfAccumulatedFrames) {
            // std::cout << "\t[Optic] \tStart\n";

            cv::cuda::cvtColor(originLeftGpuMat[0], grayPrvsLeftGpuMat, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(originRightGpuMat[0], grayPrvsRightGpuMat, cv::COLOR_BGR2GRAY);

            cv::cuda::cvtColor(originLeftGpuMat[numOfAccumulatedFrames-1], grayNextLeftGpuMat, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(originRightGpuMat[numOfAccumulatedFrames-1], grayNextRightGpuMat, cv::COLOR_BGR2GRAY);

            // Calculate optical flow
            farn->calc(grayPrvsLeftGpuMat, grayNextLeftGpuMat, _flowLeftGpuMat);
            farn->calc(grayPrvsRightGpuMat, grayNextRightGpuMat, _flowRightGpuMat);

            auto [lMean, lStd] = calcFlowMeanAndStd(_flowLeftGpuMat);
            auto [rMean, rStd] = calcFlowMeanAndStd(_flowRightGpuMat);
            
            auto [lMin, lMax] = calcFlowMinMax(_flowLeftGpuMat);
            auto [rMin, rMax] = calcFlowMinMax(_flowRightGpuMat);
            
            threadLockMutex.lock();

            if (lMean > 1 or rMean > 1) {
                opticalDetectedFlag = true;
                // std::cout << "\t[Optic] Optical flow detected!\n";

                //threadLockMutex.unlock();
                //std::this_thread::sleep_for(std::chrono::milliseconds(5));
                recording = true;
            } else {
                //threadLockMutex.unlock();
                //std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (recording == true) {
                    fileCount ++;
                    recording = false;
                }
                opticalDetectedFlag = false;
                // std::cout << "\t[Optic] Optical No!\n";

                //threadLockMutex.unlock();
                //std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            originLeftGpuMat.clear();
            originRightGpuMat.clear();
        }
    }
}

void OpticalFlowManager::calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat) {
    //cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
    farn->calc(prvsGpuMat, nextGpuMat, xyVelocityGpuMat);
}