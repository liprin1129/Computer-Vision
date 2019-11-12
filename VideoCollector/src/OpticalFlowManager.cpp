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
    cv::cuda::GpuMat &cvLeftGpuMat, cv::cuda::GpuMat &cvRightGpuMat, 
    char &key, bool &opticalDetectedFlag, sl::ERROR_CODE &grabErrorCode,
    int numOfAccumulatedFrames) {

    //std::vector<cv::cuda::GpuMat> leftGpuMatSequence(numOfAccumulatedFrames), rightGpuMatSequence(numOfAccumulatedFrames);
    int vectorCounter = 0;
    int boolCounter = 0;

    //cv::cuda::GpuMat leftGpuMatSequence[numOfAccumulatedFrames], rightGpuMatSequence[numOfAccumulatedFrames];

    while (key!='q') {
        threadLockMutex.lock();
        if (grabErrorCode == sl::SUCCESS and !cvLeftGpuMat.empty() and !cvRightGpuMat.empty()) {
            if (leftGpuMatSequence.size() < numOfAccumulatedFrames and rightGpuMatSequence.size() < numOfAccumulatedFrames) {
                cv::cuda::cvtColor(cvLeftGpuMat, grayLeftGpuMat, cv::COLOR_BGR2GRAY);
                cv::cuda::cvtColor(cvRightGpuMat, grayRightGpuMat, cv::COLOR_BGR2GRAY);

                cv::cuda::resize(grayLeftGpuMat, resizedLeftGpuMat, cv::Size(480, 270));
                cv::cuda::resize(grayRightGpuMat, resizedRightGpuMat, cv::Size(480, 270));

                //std::fprintf(stdout, "Vector size: %d, %d\n", static_cast<int>(leftGpuMatSequence.size()), static_cast<int>(rightGpuMatSequence.size()));
                
                leftGpuMatSequence.push_back(resizedLeftGpuMat.clone());
                rightGpuMatSequence.push_back(resizedRightGpuMat.clone());
            }

            if (leftGpuMatSequence.size() >= numOfAccumulatedFrames and rightGpuMatSequence.size() >= numOfAccumulatedFrames) {
                farn->calc(leftGpuMatSequence[0], leftGpuMatSequence[numOfAccumulatedFrames-1], _flowLeftGpuMat);
                farn->calc(rightGpuMatSequence[0], rightGpuMatSequence[numOfAccumulatedFrames-1], _flowRightGpuMat);
                
                // Clear the vectors
                leftGpuMatSequence.clear(); rightGpuMatSequence.clear();
                std::cout << "\tClear!\n";

                auto [lMean, lStd] = calcFlowMeanAndStd(_flowLeftGpuMat);
                auto [rMean, rStd] = calcFlowMeanAndStd(_flowRightGpuMat);
                
                auto [lMin, lMax] = calcFlowMinMax(_flowLeftGpuMat);
                auto [rMin, rMax] = calcFlowMinMax(_flowRightGpuMat);
                std::fprintf(stdout, "lMean: %f, Std: %f, rMean: %f, Std: %f\n", lMean, lStd, rMean, rStd);
                std::fprintf(stdout, "lMin: %f, lMax: %f, rMin: %f, rMax: %f\n\n", lMin, lMax, rMin, rMax);

                if (lMean > 1.0 and rMean > 1.0) {
                    opticalDetectedFlag = true;
                } else {
                    opticalDetectedFlag = false;
                }
            }

            //testMat = cvLeftGpuMat;
            
            /*
            std::fprintf(stdout, "Vector size: %d\n", static_cast<int>(gpuMatSequence.size()));
            if (gpuMatSequence.size() < frameRate) {
                gpuMatSequence.push_back(cvLeftGpuMat);
                //cvLeftGpuMat.release();
            }

            else {
                break;
            }*/
        }

        threadLockMutex.unlock();
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
    //testMat.release();
    std::cout << *cvLeftGpuMat.refcount << std::endl;
    
    key = 'q';
}

void OpticalFlowManager::calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat) {
    //cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
    farn->calc(prvsGpuMat, nextGpuMat, xyVelocityGpuMat);
}