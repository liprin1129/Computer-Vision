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

void OpticalFlowManager::startOpticalFlowThread(
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
            _originLeftGpuMat.assign(cvLeftGpuMatFrames.begin(), cvLeftGpuMatFrames.end());
            _originRightGpuMat.assign(cvRightGpuMatFrames.begin(), cvRightGpuMatFrames.end());

            //cvLeftGpuMatFrames.clear();
            //cvRightGpuMatFrames.clear();

            //isVectorFull = false;
            //key = 'q';
        }

        //std::cout << "\t[Optic ] vector size: " << originLeftGpuMat.size() << " : " << originRightGpuMat.size() << std::endl;
        
        threadLockMutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        if (_originLeftGpuMat.size() >= numOfAccumulatedFrames or _originRightGpuMat.size() >= numOfAccumulatedFrames) {
            // std::cout << "\t[Optic] \tStart\n";

            cv::cuda::cvtColor(_originLeftGpuMat[0], _greyPrvsLeftGpuMat, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(_originRightGpuMat[0], _greyPrvsRightGpuMat, cv::COLOR_BGR2GRAY);

            cv::cuda::cvtColor(_originLeftGpuMat[numOfAccumulatedFrames-1], _greyNextLeftGpuMat, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(_originRightGpuMat[numOfAccumulatedFrames-1], _greyNextRightGpuMat, cv::COLOR_BGR2GRAY);

            // Calculate optical flow
            farn->calc(_greyPrvsLeftGpuMat, _greyNextLeftGpuMat, _flowLeftGpuMat);
            farn->calc(_greyPrvsRightGpuMat, _greyNextRightGpuMat, _flowRightGpuMat);

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

            _originLeftGpuMat.clear();
            _originRightGpuMat.clear();
        }
    }
}

void OpticalFlowManager::startSideBySideOpticalFlowThread(
    std::mutex &threadLockMutex, 
    //cv::cuda::GpuMat &cvLeftGpuMat, cv::cuda::GpuMat &cvRightGpuMat, 
    std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames,
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
            _originSideBySideGpuMat.assign(cvSideBySideGpuMatFrames.begin(), cvSideBySideGpuMatFrames.end());

            //cvLeftGpuMatFrames.clear();
            //cvRightGpuMatFrames.clear();

            //isVectorFull = false;
            //key = 'q';
        }

        //std::cout << "\t[Optic ] vector size: " << originLeftGpuMat.size() << " : " << originRightGpuMat.size() << std::endl;
        
        threadLockMutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        if (_originSideBySideGpuMat.size() >= numOfAccumulatedFrames) {
            // std::cout << "\t[Optic] \tStart\n";

            // Convert to Gray images
            cv::cuda::cvtColor(_originSideBySideGpuMat[0], _greyPrvsSideBySideGpuMat, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(_originSideBySideGpuMat[numOfAccumulatedFrames-1], _greyNextSideBySideGpuMat, cv::COLOR_BGR2GRAY);

            // Calculate optical flow
            farn->calc(_greyPrvsSideBySideGpuMat, _greyNextSideBySideGpuMat, _flowSideBySideGpuMat);

            auto [sbsMean, sbsStd] = calcFlowMeanAndStd(_flowSideBySideGpuMat);
            auto [sbsMin, sbsMax] = calcFlowMinMax(_flowSideBySideGpuMat);
            
            threadLockMutex.lock();

            if (sbsMean > 1) {
                opticalDetectedFlag = true;
                // std::cout << "\t[Optic] Optical flow detected!\n";
                recording = true;
            } else {
                if (recording == true) {
                    fileCount ++;
                    recording = false;
                }
                opticalDetectedFlag = false;
                // std::cout << "\t[Optic] Optical No!\n";
            }

            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            _originSideBySideGpuMat.clear();
        }
    }
}

void OpticalFlowManager::startOpticalFlow(  std::mutex &threadLockMutex, 
                                            std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFramesThread,
                                            cv::Size frameSize,
                                            bool &isOpticalFlowDetected, 
                                            char &key, int fps) {
    //const int64 start = cv::getTickCount();
    std::string saveFile = "/DATASETs/OpticalFlow-Motion-Dataset/" + std::to_string(10) + ".avi";
    cv::VideoWriter vw = cv::VideoWriter(saveFile, CV_FOURCC('H','2','6','4'), fps, frameSize);

    int fourthFPS = static_cast<int>(fps/4*1);
    int thirdFPS = static_cast<int>(fps/4*2);
    int secondFPS = static_cast<int>(fps/4*3);
    int fullFPS = static_cast<int>(fps/4*4);
    
    int beginOpticFPS = 0;
    int endOpticFPS = 0;

    bool isCalOpticFPS = false;

    while( key != 'q') {
        // std::fprintf(stdout, "Optical Flow\n");
        // std::cout << "empty?: " << !_cvSideBySideGpuMatFrames.empty() << std::endl;
        threadLockMutex.lock();
        
        // _cvSideBySideGpuMatFrames = cvSideBySideGpuMatFramesThread;
        _cvSideBySideGpuMatFrames.assign(cvSideBySideGpuMatFramesThread.begin(), cvSideBySideGpuMatFramesThread.end());
        cvSideBySideGpuMatFramesThread.clear();
        threadLockMutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // std::fprintf(stdout, "_cvSideBySideGpuMatFrames size: %d\n", static_cast<int>(_cvSideBySideGpuMatFrames.size()));
        
        // std::fprintf(stdout, "halfFPS: %d\n", halfFPS);

        if (_cvSideBySideGpuMatFrames.size() == fourthFPS) {
            beginOpticFPS = 1;
            endOpticFPS = fourthFPS;
            isCalOpticFPS = true;
        } else if(_cvSideBySideGpuMatFrames.size() == thirdFPS) {
            beginOpticFPS = fourthFPS;
            endOpticFPS = thirdFPS;
            isCalOpticFPS = true;
        } else if(_cvSideBySideGpuMatFrames.size() == secondFPS) {
            beginOpticFPS = thirdFPS;
            endOpticFPS = secondFPS;
            isCalOpticFPS = true;
        } else if(_cvSideBySideGpuMatFrames.size() == fullFPS) {
            beginOpticFPS = secondFPS;
            endOpticFPS = fullFPS;
            isCalOpticFPS = true;
        }

        if (isCalOpticFPS) {
            
            // std::fprintf(stdout, "calculate FPS: %d\n", calOpticFPS);
            std::fprintf(stdout, "[%d, %d] ", beginOpticFPS-1, static_cast<int>(_cvSideBySideGpuMatFrames.size())-1);
            cv::cuda::cvtColor(_cvSideBySideGpuMatFrames[beginOpticFPS-1], _greyPrvsSideBySideGpuMat, cv::COLOR_RGB2GRAY);
            cv::cuda::cvtColor(_cvSideBySideGpuMatFrames[endOpticFPS-1], _greyNextSideBySideGpuMat, cv::COLOR_RGB2GRAY);

            // Calculate opticalflow
            farn->calc(_greyPrvsSideBySideGpuMat, _greyNextSideBySideGpuMat, _flowSideBySideGpuMat);
            auto [sbsMean, sbsStd] = calcFlowMeanAndStd(_flowSideBySideGpuMat);
            // std::fprintf(stdout, "\tMean: %f, Std: %f\n", sbsMean, sbsStd);

            if (sbsMean > 1) {
                // std::cout << "\t<Detected>\n";

                // threadLockMutex.lock();
                isOpticalFlowDetected = true;
                // threadLockMutex.unlock();
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                std::cout << "\t<No>\n";

                // threadLockMutex.lock();
                isOpticalFlowDetected = false;
                // threadLockMutex.unlock();
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            isCalOpticFPS = false;
        }

        if (isOpticalFlowDetected and _cvSideBySideGpuMatFrames.size() == fullFPS) {
            std::cout << "\t<Detected>\n";
            isOpticalFlowDetected = false;
            std::cout << "Size: " << _cvSideBySideGpuMatFrames.size() << std::endl;
            for (auto &i: _cvSideBySideGpuMatFrames) {
                // std::fprintf(stdout, "Save frame: %d, ", i);
                // std::fprintf(stdout, "\n");
                cv::Mat cpuFrame(i);
                vw.write(cpuFrame);
            }
            _cvSideBySideGpuMatFrames.clear();
        }
            /*for (auto &frame: _cvSideBySideGpuMatFrames) {
                cv::Mat cpuFrame(frame);
                vw.write(cpuFrame);
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
            */
        
        /*if (_cvSideBySideGpuMatFrames.size() > 29) {
            // Calculate optical flow
            cv::cuda::cvtColor(_cvSideBySideGpuMatFrames[static_cast<int>(_cvSideBySideGpuMatFrames.size()/15)-1], _greyPrvsSideBySideGpuMat, cv::COLOR_RGB2GRAY);
            cv::cuda::cvtColor(_cvSideBySideGpuMatFrames[_cvSideBySideGpuMatFrames.size()-1], _greyNextSideBySideGpuMat, cv::COLOR_RGB2GRAY);
            farn->calc(_greyPrvsSideBySideGpuMat, _greyNextSideBySideGpuMat, _flowSideBySideGpuMat);

            auto [sbsMean, sbsStd] = calcFlowMeanAndStd(_flowSideBySideGpuMat);
            //auto [sbsMin, sbsMax] = calcFlowMinMax(_flowSideBySideGpuMat);

            std::fprintf(stdout, "\tMean: %f, Std: %f\n", sbsMean, sbsStd);
            //std::fprintf(stdout, "\tMin: %f, Max: %f\n", sbsMin, sbsMax);

            // _cvSideBySideGpuMatFrames.clear();
            
            //const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
            //std::fprintf(stdout, "CameraManager time : %lf sec\n", timeSec);

            if (sbsMean > 1) {
                std::cout << "\t[Optic] Optical flow detected!\n";

                threadLockMutex.lock();
                isOpticalFlowDetected = true;
                threadLockMutex.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                std::cout << "\t[Optic] Optical No!\n";

                threadLockMutex.lock();
                isOpticalFlowDetected = false;
                threadLockMutex.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        if (_cvSideBySideGpuMatFrames.size() >= 59) {
            // std::cout << "\t" << cv::cuda::sum(cvSideBySideGpuMatFramesThread[0]) << " | " << cv::cuda::sum(cvSideBySideGpuMatFramesThread[1]) << std::endl;
            // _cvSideBySideGpuMatFrames.assign(cvSideBySideGpuMatFramesThread.begin(), cvSideBySideGpuMatFramesThread.end());
            _cvSideBySideGpuMatFrames.clear();
        }
        */
    }
}

void OpticalFlowManager::calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat) {
    farn->calc(prvsGpuMat, nextGpuMat, xyVelocityGpuMat);
}