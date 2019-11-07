
#include "OpticalFlowManager.h"

inline bool isFlowCorrect(cv::Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static cv::Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static cv::Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    cv::Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const cv::Mat_<float>& flowx, const cv::Mat_<float>& flowy, cv::Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                cv::Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = cv::max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            cv::Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void showFlow(const char* name, const cv::cuda::GpuMat& d_flow)
{
    cv::cuda::GpuMat planes[2];
    cv::cuda::split(d_flow, planes);

    cv::Mat flowx(planes[0]);
    cv::Mat flowy(planes[1]);

    cv::Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    cv::imshow(name, out);
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

OpticalFlowManager::OpticalFlowManager() {
    _firstGetFlag = false; _secondGetFlag = true;

    //cv::namedWindow("OriginLeftView", cv::WINDOW_OPENGL);
    //cv::namedWindow("OriginRightView", cv::WINDOW_OPENGL);
    
    //cv::namedWindow("FlowView", cv::WINDOW_OPENGL);

    //startOpticalFlow();
}

OpticalFlowManager::~OpticalFlowManager() {
    std::fprintf(stdout, "%s\n", "OpticalFlowManager destructor");
    //cv::destroyWindow("OriginLeftView");
    //cv::destroyWindow("OriginRightView");
    
    //cv::destroyWindow("FlowView");
}

template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
    x = cv::max(cv::min(x, b), a);
    return c + (d-c) * (x-a) / (b-a);
}

static void colorizeFlow(const cv::Mat &u, const cv::Mat &v, cv::Mat &dst)
{
    double uMin, uMax;
    cv::minMaxLoc(u, &uMin, &uMax, 0, 0);
    double vMin, vMax;
    cv::minMaxLoc(v, &vMin, &vMax, 0, 0);
    uMin = cv::abs(uMin); uMax = cv::abs(uMax);
    vMin = cv::abs(vMin); vMax = cv::abs(vMax);
    float dMax = static_cast<float>(cv::max(cv::max(uMin, uMax), cv::max(vMin, vMax)));

    dst.create(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            dst.at<uchar>(y,3*x) = 0;
            dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
            dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
        }
    }
}

void OpticalFlowManager::startOpticalFlow() {
    cv::cuda::GpuMat resizedPrvsLeftGpuMat, resizedPrvsRightGpuMat;
    cv::cuda::GpuMat resizedNextLeftGpuMat, resizedNextRightGpuMat;

    char key = ' ';
    while (key != 'q') {
        
        // Save the first frame
        if(_firstGetFlag == false and _secondGetFlag == true and getOneFrameFromZED() == sl::SUCCESS) {
            cv::cuda::cvtColor(_cvRightGpuMat, _prvsRightGpuMat, cv::COLOR_BGR2GRAY);
            
            //std::fprintf(stdout, "Get first\n");
            //std::fprintf(stdout, "cvRightGpuMat @: %d,\t prvsRightGpuMat @: %d\n", *_cvRightGpuMat.refcount, *_prvsRightGpuMat.refcount);

            cv::cuda::resize(_prvsRightGpuMat, resizedPrvsRightGpuMat, cv::Size(480, 270));
            _firstGetFlag = true; _secondGetFlag = false;
        }

        // Save the second frame
        if(_firstGetFlag == true and _secondGetFlag == false and getOneFrameFromZED() == sl::SUCCESS) {
            cv::cuda::cvtColor(_cvRightGpuMat, _nextRightGpuMat, cv::COLOR_BGR2GRAY);
            //std::fprintf(stdout, "\tGet second\n");
            //std::fprintf(stdout, "cvRightGpuMat @: %d,\t prvsRightGpuMat @: %d\n", *_cvRightGpuMat.refcount, *_prvsRightGpuMat.refcount);

            cv::cuda::resize(_prvsRightGpuMat, resizedNextRightGpuMat, cv::Size(480, 270));
            _firstGetFlag = false; _secondGetFlag = true;
        }
        
        /*
        if (!_prvsRightGpuMat.empty() and !_nextRightGpuMat.empty()) {
            cv::imshow("OriginRightView", resizedPrvsRightGpuMat);
            cv::imshow("OriginLeftView", resizedNextRightGpuMat);
            key = cv::waitKey(10);
        }
        */

        
        //cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
        //cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();
        {
            const int64 start = cv::getTickCount();

            //farn->calc(resizedPrvsRightGpuMat, resizedNextRightGpuMat, _flowGpuMat);
            //tvl1->calc(prvsRightGpuMat, nextRightGpuMat, flowGpuMat);
            calcOpticalFlowGPU(resizedPrvsRightGpuMat, resizedNextRightGpuMat, _flowGpuMat);
            const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();

            auto [mean, std] = calcFlowMagnitude(_flowGpuMat);

            std::fprintf(stdout, "FlowCalc time : %lf sec\n", timeSec);

            std::fprintf(stdout, "Maen: %f, Std: %f\n", mean, std);
            //showFlow("FlowView", flowGpuMat);
        }
    }
}

void OpticalFlowManager::calcOpticalFlowGPU(cv::cuda::GpuMat &prvsGpuMat, cv::cuda::GpuMat &nextGpuMat, cv::cuda::GpuMat &xyVelocityGpuMat) {
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
    farn->calc(prvsGpuMat, nextGpuMat, _flowGpuMat);
}