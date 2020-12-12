#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Driver.h"
#include "Sobel_CUDA.h"
#include "Gaussian_Blur.h"
#include "Noise_Addition_CUDA.h"
#include "Gaussian_Blur_Seperated.h"
#include "Mean_Blur_Seperated.h"
#include "Canny_CUDA.h"
#include "sharpen_CUDA.h"
#include "bokeh_blur.h"

using namespace std;
using namespace cv;

void video_related(int task);
void photo_related(int task);

int main()
{
	printf("Press : \n0 for mean blur on camera input\n");
	printf("1 for sobel filter on camera input\n");
	printf("2 for Salt and pepper noise addition on camera input\n");
	printf("3 for noise reduction on camera input\n");
	printf("4 for Gaussian Blur on camera input\n");
	printf("5 for Sharpening on camera input\n");
	printf("6 for sobel filter on a photo\n");
	printf("7 for Salt and pepper noise addition on a photo\n");
	printf("8 for noise reduction on a photo\n");
	printf("9 for Gaussian Blur on a photo\n");
	printf("10 for Sharpening on a photo\n");
	printf("11 for Canny on a photo\n");
	printf("12 for Mean Blur on a photo\n");
	printf("13 for Bokeh Blur on a photo\n");

	int x;
	cin >> x;
	if (x < 6)
	{
		video_related(x);
	}
	else
	{
		photo_related(x);
	}
}

void video_related(int task)
{
	//open the video file for reading
	VideoCapture cap(0);

	// if not success, exit program
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video file" << endl;
		cin.get(); //wait for any key press
		return;
	}

	//get the frames rate of the video
	double fps = cap.get(CAP_PROP_FPS);

	int x = 0;
	
	while (true)
	{
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video
		//cout << frame.rows << " " << frame.cols << endl;
		//Breaking the while loop at the end of the video
		if (bSuccess == false)
		{
			cout << "Stream ended :(" << endl;
			break;
		}

		int height = frame.rows;
		int width = frame.cols;

		//splitting into channels
		vector<Mat> channels(3);
		split(frame, channels);

		//converting image to gray scale if needed. we'll either use this or channels depending on task
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		
		imshow("Original image", frame);
		
		Mat display;
		switch (task)
		{
		case 0:
			Mean_Blur_Seperated(channels[0].data, channels[0].rows, channels[0].cols);
			Mean_Blur_Seperated(channels[1].data, channels[1].rows, channels[1].cols);
			Mean_Blur_Seperated(channels[2].data, channels[2].rows, channels[2].cols);
			merge(channels, display);
			imshow("Mean Blurred", display);
			break;
		case 1:
			Sobel_CUDA(gray.data, gray.rows, gray.cols);
			imshow("Sobel Output", gray);
			break;
		case 2:
			Salt_Pepper(channels[0].data, channels[0].rows, channels[0].cols);
			Salt_Pepper(channels[1].data, channels[1].rows, channels[1].cols);
			Salt_Pepper(channels[2].data, channels[2].rows, channels[2].cols);
			merge(channels, display);
			imshow("Added Noise", display);
			break;
		case 3:
			Gaussian_Blur_Seperated(channels[0].data, channels[0].rows, channels[0].cols);
			Gaussian_Blur_Seperated(channels[1].data, channels[1].rows, channels[1].cols);
			Gaussian_Blur_Seperated(channels[2].data, channels[2].rows, channels[2].cols);
			Sharpen_CUDA(channels[0].data, channels[0].rows, channels[0].cols);
			Sharpen_CUDA(channels[1].data, channels[1].rows, channels[1].cols);
			Sharpen_CUDA(channels[2].data, channels[2].rows, channels[2].cols);
			merge(channels, display);
			imshow("Reduced Noise", display);
			break;
		case 4:
			Gaussian_Blur_Seperated(channels[0].data, channels[0].rows, channels[0].cols);
			Gaussian_Blur_Seperated(channels[1].data, channels[1].rows, channels[1].cols);
			Gaussian_Blur_Seperated(channels[2].data, channels[2].rows, channels[2].cols);
			merge(channels, display);
			imshow("Gaussian Blur", display);
			break;
		case 5:
			//Sharpen_CUDA(channels[0].data, channels[0].rows, channels[0].cols);
			//Sharpen_CUDA(channels[1].data, channels[1].rows, channels[1].cols);
			//Sharpen_CUDA(channels[2].data, channels[2].rows, channels[2].cols);
			//merge(channels, display);
			//imshow("Reduced Noise", display);
			//break;
			Mat image = Mat::zeros(30, 30, CV_8UC1);
			circle(image, Point(7, 7), 7, Scalar(255, 255, 255), -1);
			imshow("image", image);
			Bokeh_Blur_CUDA(channels[0].data , channels[0].rows, channels[0].cols , image.data, image.rows, image.cols);
			Bokeh_Blur_CUDA(channels[1].data, channels[1].rows, channels[1].cols , image.data, image.rows, image.cols);
			Bokeh_Blur_CUDA(channels[2].data,channels[2].rows, channels[2].cols , image.data, image.rows, image.cols);
			merge(channels, display);
			imshow("Mean Blurred", display);
			break;
		}
		

		
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}
}

void photo_related(int task)
{
	printf("Enter image path : ");

	//String image_path;
	String image_path = "C:\\Users\\DevanshuChaudhari\\Desktop\\convolve\\christmas.jpg";
	//cin >> image_path;

	Mat img = imread(image_path, IMREAD_COLOR);
	
	if (img.empty())
	{
		cout << "Could not read the image: " << image_path << std::endl;
		return;
	}

	vector<Mat> channels(3);
	split(img, channels);

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	imshow("Original image", img);

	Mat display;
	switch (task)
	{
	case 6:
		Sobel_CUDA(gray.data, gray.rows, gray.cols);
		imshow("Sobel Output", gray);
		break;
	case 7:
		Salt_Pepper(channels[0].data, channels[0].rows, channels[0].cols);
		Salt_Pepper(channels[1].data, channels[1].rows, channels[1].cols);
		Salt_Pepper(channels[2].data, channels[2].rows, channels[2].cols);
		merge(channels, display);
		imshow("Added Noise", display);
		break;
	case 8:
		Gaussian_Blur_Seperated(channels[0].data, channels[0].rows, channels[0].cols);
		Gaussian_Blur_Seperated(channels[1].data, channels[1].rows, channels[1].cols);
		Gaussian_Blur_Seperated(channels[2].data, channels[2].rows, channels[2].cols);
		Sharpen_CUDA(channels[0].data, channels[0].rows, channels[0].cols);
		Sharpen_CUDA(channels[1].data, channels[1].rows, channels[1].cols);
		Sharpen_CUDA(channels[2].data, channels[2].rows, channels[2].cols);
		merge(channels, display);
		imshow("Reduced Noise", display);
		break;
	case 9:
		Gaussian_Blur_Seperated(channels[0].data, channels[0].rows, channels[0].cols);
		Gaussian_Blur_Seperated(channels[1].data, channels[1].rows, channels[1].cols);
		Gaussian_Blur_Seperated(channels[2].data, channels[2].rows, channels[2].cols);
		merge(channels, display);
		imshow("Gaussian Blur", display);
		break;
	case 10:
		Sharpen_CUDA(channels[0].data, channels[0].rows, channels[0].cols);
		Sharpen_CUDA(channels[1].data, channels[1].rows, channels[1].cols);
		Sharpen_CUDA(channels[2].data, channels[2].rows, channels[2].cols);
		merge(channels, display);
		imshow("Reduced Noise", display);
		break;
	case 11:
		Gaussian_Blur_CUDA(gray.data, gray.rows, gray.cols);
		Sharpen_CUDA(gray.data, gray.rows, gray.cols);
		Sobel_CUDA(gray.data, gray.rows, gray.cols);
		Canny_CUDA(gray.data, gray.rows, gray.cols);
		imshow("Canny Output", gray);
		break;
	case 12:
		Mean_Blur_Seperated(channels[0].data, channels[0].rows, channels[0].cols);
		Mean_Blur_Seperated(channels[1].data, channels[1].rows, channels[1].cols);
		Mean_Blur_Seperated(channels[2].data, channels[2].rows, channels[2].cols);
		merge(channels, display);
		imshow("Mean Blurred", display);

		break;
	case 13:
		Mat image = Mat::zeros(32, 32, CV_8UC1);
		circle(image, Point(16 ,16 ), 16, Scalar(255, 255, 255), -1);
		imshow("image", image);
		Bokeh_Blur_CUDA(channels[0].data, channels[0].rows, channels[0].cols, image.data, image.rows, image.cols);
		Bokeh_Blur_CUDA(channels[1].data, channels[1].rows, channels[1].cols, image.data, image.rows, image.cols);
		Bokeh_Blur_CUDA(channels[2].data, channels[2].rows, channels[2].cols, image.data, image.rows, image.cols);
		merge(channels, display);
		imshow("Bokeh Blurred", display);
		break;
	}

	int k = waitKey(0); // Wait for a keystroke in the window
	return;
}