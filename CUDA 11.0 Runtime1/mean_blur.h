#pragma once
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace cv;
using namespace std;

void mean_blur(unsigned char* img, int width, int height)
{
	double mean_blur_filter[3][3] = { {1,1,1},{1,1,1},{1,1,1} };
	double val = 0;
	unsigned char* padded_img = (unsigned char*)malloc((long long)(width + 2) * (height + 2) * sizeof(unsigned char));
	for (int i = 1; i <= height; i++)
		for (int j = 1; j <= width; j++)
			padded_img[(width + 2) * i + j] = img[width * (i - 1) + (j - 1)];
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			val = 0;
			for (int l = -1; l <= 1; l++)
				for (int r = -1; r <= 1; r++)
					val += padded_img[(width + 2) * (i + 1 + l) + (j + 1 + r)] * mean_blur_filter[l + 1][r + 1];
			val /= 9;
			img[width * i + j] = val;
		}
}