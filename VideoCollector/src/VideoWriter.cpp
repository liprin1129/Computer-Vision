#include "VideoWriter.h"

VideoWriter::VideoWriter() {
    fileCount = 0;
    recording = false;
    std::fprintf(stdout, "VideoWiter Constructor\n");
}


void VideoWriter::writeCvGpuFramesToVideoFormat(
    std::mutex &threadLockMutex, char &key, 
    std::string fileName, int numFrames, cv::Size frameSize, 
    std::vector<cv::cuda::GpuMat> &cvLeftGpuMatFrames, std::vector<cv::cuda::GpuMat> &cvRightGpuMatFrames, 
    int &numOfFiles, bool &opticalFlowDetectedFlag, bool &isVectorFull) 
    {
        cv::Ptr<cv::cudacodec::VideoWriter> gpuWriter;
        int currentFileCount = numOfFiles;

        while (key != 'q') {

            //const int64 start = cv::getTickCount();

            /* Copy thread shared frames instnace to class frames instance */

            threadLockMutex.lock();
            //std::cout << "\t\t[Writer] Vector: " << cvLeftGpuMatFrames.size() << ", " << cvRightGpuMatFrames.size() << std::endl;
            //std::fprintf(stdout, "\t\t[Writer] Trigger?: %s\n", opticalFlowDetectedFlag? "True":"False");

            if (opticalFlowDetectedFlag) {
                //std::cout << "\t\t[Writer] True\n";

                _originLeftGpuMat.assign(cvLeftGpuMatFrames.begin(), cvLeftGpuMatFrames.end());
                _originRightGpuMat.assign(cvRightGpuMatFrames.begin(), cvRightGpuMatFrames.end());

                cvLeftGpuMatFrames.clear();
                cvRightGpuMatFrames.clear();

                //std::cout << cvLeftGpuMatFrames.size() << ", " << cvRightGpuMatFrames.size() << std::endl;

                // isOpticTriggered = opticalFlowDetectedFlag;
                // std::fprintf(stdout, "\t\t[Writer] Orig Trigger?: %s\n", opticalFlowDetectedFlag? "True":"False");
            }
                        
            //std::fprintf(stdout, "\t\t[Writer] Trigger?: %s\n", isOpticTriggered? "True":"False");

            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            
            if (_originLeftGpuMat.size() >= numFrames or _originRightGpuMat.size() > numFrames) {
                //std::cout << currentFileCount << " : " << numOfFiles << std::endl;
                if (currentFileCount == numOfFiles) {
                    toSavePath = fileName + std::to_string(numOfFiles + fileCount) + ".avi";
                    std::cout << toSavePath << std::endl;

                    vw = cv::VideoWriter(toSavePath, CV_FOURCC('H','2','6','4'), numFrames, frameSize);
                    currentFileCount++;
                }

                for (auto i: _originLeftGpuMat) {
                    i.download(_originLeftCpuMat);
                    vw.write(_originLeftCpuMat);
                }

                _originLeftGpuMat.clear(); _originRightGpuMat.clear();
            }

            //const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
            //std::fprintf(stdout, "\t\tVideoWriter time : %lf sec\n", timeSec);
        }
        vw.release();
    }

void VideoWriter::writeSideBySideCvGpuFramesToVideoFormat(
    std::mutex &threadLockMutex, char &key, 
    std::string fileName, int numFrames, cv::Size frameSize, 
    std::vector<cv::cuda::GpuMat> &cvSideBySideGpuMatFrames, 
    int &numOfFiles, bool &opticalFlowDetectedFlag, bool &isVectorFull) 
    {
        cv::Ptr<cv::cudacodec::VideoWriter> gpuWriter;
        int currentFileCount = numOfFiles;
        //std::cout << "Original frame size: " << frameSize << std::endl;

        while (key != 'q') {

            //const int64 start = cv::getTickCount();

            /* Copy thread shared frames instnace to class frames instance */

            threadLockMutex.lock();
            //std::cout << "\t\t[Writer] Vector: " << cvLeftGpuMatFrames.size() << ", " << cvRightGpuMatFrames.size() << std::endl;
            //std::fprintf(stdout, "\t\t[Writer] Trigger?: %s\n", opticalFlowDetectedFlag? "True":"False");

            if (opticalFlowDetectedFlag) {
                //std::cout << "\t\t[Writer] True\n";

                _originSideBySideGpuMat.assign(cvSideBySideGpuMatFrames.begin(), cvSideBySideGpuMatFrames.end());

                cvSideBySideGpuMatFrames.clear();

                //std::cout << cvLeftGpuMatFrames.size() << ", " << cvRightGpuMatFrames.size() << std::endl;

                // isOpticTriggered = opticalFlowDetectedFlag;
                // std::fprintf(stdout, "\t\t[Writer] Orig Trigger?: %s\n", opticalFlowDetectedFlag? "True":"False");
            }
                        
            //std::fprintf(stdout, "\t\t[Writer] Trigger?: %s\n", isOpticTriggered? "True":"False");

            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            
            if (_originSideBySideGpuMat.size() >= numFrames) {
                //std::cout << currentFileCount << " : " << numOfFiles << std::endl;
                if (currentFileCount == numOfFiles) {
                    toSavePath = fileName + std::to_string(numOfFiles + fileCount) + ".avi";
                    std::cout << toSavePath << std::endl;

                    vw = cv::VideoWriter(toSavePath, CV_FOURCC('H','2','6','4'), numFrames, frameSize);
                    //vw = cv::VideoWriter(toSavePath, CV_FOURCC('H','2','6','4'), 60, cv::Size(1280, 720));
                    //vw = cv::VideoWriter(toSavePath, CV_FOURCC('M','J','P','G'), 60, cv::Size(1280, 720));
                    //std::fprintf(stdout, "File count: %d\n", currentFileCount);
                    currentFileCount++;

                    //std::cout << "Frame Size: " << _originSideBySideGpuMat[0].size() << std::endl;
                }

                //int k = 0;
                for (auto i: _originSideBySideGpuMat) {
                    //std::fprintf(stdout, "Frame[%d]: ", k);
                    //std::fprintf(stdout, "File count: %d\n", currentFileCount);
                    //std::cout << _originSideBySideGpuMat[k].size() << std::endl;
                    //k++;

                    i.download(_originSideBySideCpuMat);
                    vw.write(_originSideBySideCpuMat);
                    //std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    //vw.write(_originSideBySideCpuMat(cv::Rect(0, 0, 1280, 720)));
                    //std::cout << "Video write\n";
                }

                _originSideBySideGpuMat.clear();
            }

            //const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
            //std::fprintf(stdout, "\t\tVideoWriter time : %lf sec\n", timeSec);
        }
        vw.release();
    }

void VideoWriter::writeSlFramesToVideoFormat(
    std::mutex &threadLockMutex, char &key,
    std::string fileName, int numFrames, cv::Size frameSize, 
    sl::Camera &zed,
    int &numOfFiles, bool &opticalFlowDetectedFlag) 
    {
        while (key != 'q') {
            threadLockMutex.lock();

            if (opticalFlowDetectedFlag) {
                std::cout << "Save..." << std::endl;
                if (!recording) {
                    fileCount ++;
                    recording = true;
                    
                    std::string toSavePath = fileName + std::to_string(numOfFiles + fileCount) + ".svo";
                    const char *toSaveFileName = toSavePath.c_str();
                    std::cout << toSaveFileName << std::endl;

                    zed.enableRecording(sl::String(toSaveFileName), sl::SVO_COMPRESSION_MODE_LOSSLESS);
                }

                zed.record();
            }

            else {
                if (recording) {
                    std::cout << "Save disable." << std::endl;
                    zed.disableRecording();
                }
                recording = false;
            }
            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::nanoseconds(10));
        }
    }