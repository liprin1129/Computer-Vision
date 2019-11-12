#include "VideoWriter.h"

VideoWriter::VideoWriter() {
    fileCount = 0;
    recording = false;
    std::fprintf(stdout, "VideoWiter Constructor\n");
}

void VideoWriter::writeFramesToVideoFormat(
    std::mutex &threadLockMutex, char &key, 
    std::string fileName, int numFrames, cv::Size frameSize, 
    cv::cuda::GpuMat &originGpuMat,
    int &numOfFiles, bool &opticalFlowDetectedFlag) {
        
        //vw = cv::VideoWriter("/DATASETs/OpticalFlow-Motion-Dataset/1.avi", CV_FOURCC('M','J','P','G'), numFrames, frameSize);
        vw = cv::VideoWriter("/DATASETs/OpticalFlow-Motion-Dataset/1.avi", CV_FOURCC('X','2','6','4'), numFrames, frameSize);
        
        while (key != 'q') {
            /*
            threadLockMutex.lock();

            if (opticalFlowDetectedFlag) {
                
                if (!recording) {
                    fileCount ++;
                    recording = true;

                    toSavePath = fileName + std::to_string(numOfFiles + fileCount) + ".avi";
                    
                    //vw = cv::VideoWriter("/DATASETs/OpticalFlow-Motion-Dataset/1.avi", CV_FOURCC('M','J','P','G'), numFrames, frameSize);
                    vw = cv::VideoWriter(toSavePath, CV_FOURCC('M','J','P','G'), numFrames, frameSize);
                    //std::cout << "Recording\n";
                }
                std::cout << toSavePath << std::endl;
                //std::cout << ".";
                
                cv::Mat originCpuMat;
                originGpuMat.download(originCpuMat);
                //std::this_thread::sleep_for(std::chrono::milliseconds(500));
                vw.write(originCpuMat);
            }
            else {
                recording = false;
            }

            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            */
            if (!originGpuMat.empty()) {
                //threadLockMutex.lock();
                
                originGpuMat.download(originCpuMat);
                vw.write(originCpuMat);
                //threadLockMutex.unlock();
                //std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            }
        }

        vw.release();
}