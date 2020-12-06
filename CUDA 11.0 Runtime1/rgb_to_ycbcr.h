#ifndef _rgb_to_ycbcr_
#define _rgb_to_ycbcr_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
Mat rgb2ycbcr(Mat input_image);
Mat ycbcr2rgb(Mat input_image);
#endif