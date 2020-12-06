#include "entrypoint.h"
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rgb_to_ycbcr.h"

using namespace std;
using namespace cv;

void compress()
{
    String image_path = "D:\\GitHub\\CV_CUDA\\CUDA 11.0 Runtime1\\CUDA 11.0 Runtime1\\images\\input_image_1.jpg";
    //String image_path =  samples::findFile("input_images_1.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Could not read the image: " << image_path << std::endl;
        return;
    }
    
    Mat result = rgb2ycbcr(img);
    cout << "rows = " << result.rows << " columns = " << result.cols;

    Mat result2 = ycbcr2rgb(result);

    Mat display = Mat();
    // Scaling the Image
    cv::resize(result2, display, cv::Size(), 0.25, 0.25);

    imshow("Display window", display);
    int k = waitKey(0); // Wait for a keystroke in the window
}