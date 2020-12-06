#ifndef _updownsample_
#define _updownsample_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
Mat upsample(Mat input_image);
Mat downsample(Mat input_image);
#endif
