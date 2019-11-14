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

            const int64 start = cv::getTickCount();

            /* Copy thread shared frames instnace to class frames instance */

            threadLockMutex.lock();
            //std::cout << "\t\t[Writer] Vector: " << cvLeftGpuMatFrames.size() << ", " << cvRightGpuMatFrames.size() << std::endl;
            //std::fprintf(stdout, "\t\t[Writer] Trigger?: %s\n", opticalFlowDetectedFlag? "True":"False");

            if (opticalFlowDetectedFlag) {
                //std::cout << "\t\t[Writer] True\n";

                originLeftGpuMat.assign(cvLeftGpuMatFrames.begin(), cvLeftGpuMatFrames.end());
                originRightGpuMat.assign(cvRightGpuMatFrames.begin(), cvRightGpuMatFrames.end());

                cvLeftGpuMatFrames.clear();
                cvRightGpuMatFrames.clear();

                //std::cout << cvLeftGpuMatFrames.size() << ", " << cvRightGpuMatFrames.size() << std::endl;

                // isOpticTriggered = opticalFlowDetectedFlag;
                // std::fprintf(stdout, "\t\t[Writer] Orig Trigger?: %s\n", opticalFlowDetectedFlag? "True":"False");
            }
                        
            //std::fprintf(stdout, "\t\t[Writer] Trigger?: %s\n", isOpticTriggered? "True":"False");

            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            
            if (originLeftGpuMat.size() >= numFrames or originRightGpuMat.size() > numFrames) {
                //std::cout << currentFileCount << " : " << numOfFiles << std::endl;
                if (currentFileCount == numOfFiles) {
                    toSavePath = fileName + std::to_string(numOfFiles + fileCount) + ".avi";
                    std::cout << toSavePath << std::endl;

                    vw = cv::VideoWriter(toSavePath, CV_FOURCC('H','2','6','4'), numFrames, frameSize);
                    currentFileCount++;
                }

                for (auto i: originLeftGpuMat) {
                    i.download(originLeftCpuMat);
                    vw.write(originLeftCpuMat);
                }

                originLeftGpuMat.clear(); originRightGpuMat.clear();
            }

            const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
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