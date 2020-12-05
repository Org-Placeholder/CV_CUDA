#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rgb_to_ycbcr.h"
#include <iostream>
#include <stdio.h>

using namespace cv;

__global__ void rgb2ycbcr_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);

__global__ void ycbcr2rgb_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);

Mat rgb2ycbcr(Mat input_image)
{
	Mat channels[3];
	split(input_image, channels);
	return channels[1];
}

void ycbcr2rgb(Mat input_image)
{

}

__global__ void rgb2ycbcr_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{

}

__global__ void ycbcr2rgb_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{

}