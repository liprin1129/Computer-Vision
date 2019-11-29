/**
 * C++ standard library
 */
#include <iostream>

/**
 * ZED SDK
 */
#include <sl/Camera.hpp>
#include <sl/Core.hpp>

/**
 * OpenCV
 */
#include <opencv2/opencv.hpp>

/**
 * For thread usage
 */
#include <thread> // std::this_thread::sleep_for
#include <chrono> // std::chrono::seconds
#include <mutex> // std::mutex for lock shared variable