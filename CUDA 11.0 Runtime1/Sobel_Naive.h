#pragma once
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;

void sobel_naive(unsigned char* img, int width, int height)
{
	int sobel_filterx[3][3] = { 1,0,-1,2,0,-2,1,0,-1 };
	int sobel_filtery[3][3] = { 1,2,1,0,0,0,-1,-2,-1 };
	int hor, ver;
	unsigned char* padded_img = (unsigned char*)malloc((long long)(width + 2) * (height + 2) * sizeof(unsigned char));
	for (int i = 1; i <= height; i++)
		for (int j = 1; j <= width; j++)
			padded_img[(width + 2) * i + j] = img[width * (i - 1) + (j - 1)];
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			hor = ver = 0;
			for (int l = -1; l <= 1; l++)
				for (int r = -1; r <= 1; r++)
				{
					hor += padded_img[(width + 2) * (i + 1 + l) + (j + 1 + r)] * sobel_filterx[l + 1][r + 1];
					ver += padded_img[(width + 2) * (i + 1 + l) + (j + 1 + r)] * sobel_filtery[l + 1][r + 1];
				}
			img[width * i + j] = sqrt(hor * hor + ver * ver) / 4;
		}
}
