#pragma once
#include<opencv2/opencv.hpp>
#include "Sharpen.h"
#include "Gaussian_Blur.h"
#include "Sobel_Naive.h"

using namespace cv;
using namespace std;

void non_maximum_suppression_without_interpolation(unsigned char* img, int width, int height)
{
	double* angle = (double*)malloc(height * width * sizeof(double));
	sobel_naive(img, width, height, angle);
	unsigned char* padded_img = (unsigned char*)malloc((width + 2) * (height + 2) * sizeof(unsigned char));
	for (int i = 1; i <= height; i++)
		for (int j = 1; j <= width; j++)
			padded_img[i * (width + 2) + j] = img[(i - 1) * width + (j - 1)];
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			int x = padded_img[(i + 1) * (width + 2) + (j + 1)];
			double ang = angle[i * width + j];
			if ((ang >= -22.5 && ang <= 22.5) || (ang >= -180 && ang < -157.5)) // (1) (i,j+1) & (i,j-1)
				if (x > padded_img[(i + 1) * (width + 2) + j] && x > padded_img[(i + 1) * (width + 2) + (j + 2)])
					img[i * width + j] = x;
				else
					img[i * width + j] = 0;
			else if ((ang >= 22.5 && ang <= 67.5) || (ang < -112.5 && ang >= -157.5)) // (2)  (i-1,j+1) & (i+1,j-1)
				if (x > padded_img[i * (width + 2) + j + 2] && x > padded_img[(i + 2) * (width + 2) + j])
					img[i * width + j] = x;
				else
					img[i * width + j] = 0;
			else if ((ang >= 67.5 && ang < 112.5) || (ang < -67.5 && ang >= -112.5)) // (3)  (i+1,j) & (i-1,j)
				if (x > padded_img[(i + 2) * (width + 2) + (j + 1)] && x > padded_img[i * (width + 2) + j + 1])
					img[i * width + j] = x;
				else
					img[i * width + j] = 0;
			else                                                               // (4)  (i+1,j+1) & (i-1,j-1)
				if (x > padded_img[(i + 2) * (width + 2) + j + 2] && x > padded_img[i * (width + 2) + j])
					img[i * width + j] = x;
				else
					img[i * width + j] = 0;
		}
}
void double_thresholding(unsigned char* img, int width, int height, double threshold_val)
{
	threshold_val *= 255;
	int threshold_pixel = threshold_val;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (img[i * width + j] > threshold_pixel)
				img[i * width + j] = 255;
			else
				img[i * width + j] = 0;
}
void canny(unsigned char* img, int width, int height)
{
	gaussian_blur(img, width, height); // for noise reduction
	sharpen(img, width, height);      // noise reduction
	non_maximum_suppression_without_interpolation(img, width, height);
	double_thresholding(img, width, height, 0.4); // for enhancing edge quality
}
/*
    (3)
.   .   . (2)

.   .   . (1)

.   .   .  (4)
*/
