#include "VideoWriter.h"

VideoWriter::VideoWriter() {
    std::fprintf(stdout, "VideoWiter Constructor\n");
}

void VideoWriter::writeFramesToVideoFormat(
    std::mutex &threadLockMutex, char &key, 
    std::string fileName, int numFrames, cv::Size frameSize, 
    cv::cuda::GpuMat &originGpuMat) {

    vw = cv::VideoWriter(fileName, CV_FOURCC('M','J','P','G'), numFrames, frameSize);

    while (key != 'q') {
        if (!originGpuMat.empty()) {
            std::cout << "Writer.\n";

            threadLockMutex.lock();

            cv::Mat originCpuMat;
            originGpuMat.download(originCpuMat);
            vw.write(originCpuMat);

            threadLockMutex.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

    vw.release();
}