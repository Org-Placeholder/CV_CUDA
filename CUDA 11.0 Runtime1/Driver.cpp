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

using namespace std;
using namespace cv;

int main()
{

	//open the video file for reading
	//VideoCapture cap(0);

	// if not success, exit program
	//if (cap.isOpened() == false)
	//{
	//	cout << "Cannot open the video file" << endl;
	//	cin.get(); //wait for any key press
	//	return -1;
	//}

	//get the frames rate of the video
	//double fps = cap.get(CAP_PROP_FPS);
	//cout << "Frames per seconds : " << fps << endl;

	String window_name = "CV Visualization";

	//namedWindow(window_name, WINDOW_NORMAL); //create a window
	int x = 0;
	/*
	while (true)
	{
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video 

		//Breaking the while loop at the end of the video
		if (bSuccess == false)
		{
			cout << "Stream ended :(" << endl;
			break;
		}

		int height = frame.rows;
		int width = frame.cols;

		//converting image to gray scale
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		
		//Function call
		
		//Salt_Pepper(gray.data, gray.rows, gray.cols);
		//imshow("noise", gray);
		Gaussian_Blur_CUDA(gray.data, gray.rows, gray.cols);
		//Mean_Blur_Seperated(gray.data, gray.rows, gray.cols);
		//Image display
		Sharpen_CUDA(gray.data, gray.rows, gray.cols);
		Sobel_CUDA(gray.data, gray.rows, gray.cols);
		imshow("original", gray);
		Canny_CUDA(gray.data, gray.rows, gray.cols);
		imshow(window_name, gray);

		//wait for for 10 ms until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If the any other key is pressed, continue the loop 
		//If any key is not pressed withing 10 ms, continue the loop
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}*/
	String image_path = "D:\\GitHub\\CV_CUDA\\CUDA 11.0 Runtime1\\CUDA 11.0 Runtime1\\images\\input_image_3.jpg";
	//String image_path =  samples::findFile("input_images_1.jpg");
	Mat img = imread(image_path, IMREAD_COLOR);
	if (img.empty())
	{
		cout << "Could not read the image: " << image_path << std::endl;
		return 0;
	}
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	//Salt_Pepper(gray.data, gray.rows, gray.cols);
		//imshow("noise", gray);
	Gaussian_Blur_CUDA(gray.data, gray.rows, gray.cols);
	//Gaussian_Blur_CUDA(gray.data, gray.rows, gray.cols);
	//Mean_Blur_Seperated(gray.data, gray.rows, gray.cols);
	//Image display
	Sharpen_CUDA(gray.data, gray.rows, gray.cols);
	Sobel_CUDA(gray.data, gray.rows, gray.cols);
	imshow("original", gray);
	Canny_CUDA(gray.data, gray.rows, gray.cols);
	imshow("Display window", gray);
	int k = waitKey(0); // Wait for a keystroke in the window
	return 0;
}