#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Driver.h"
#include "Sobel_CUDA.h"
#include "Gaussian_Blur.h"

using namespace std;
using namespace cv;

int main()
{

	//open the video file for reading
	VideoCapture cap(0);

	// if not success, exit program
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video file" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	//get the frames rate of the video
	double fps = cap.get(CAP_PROP_FPS);
	cout << "Frames per seconds : " << fps << endl;

	String window_name = "CV Visualization";

	namedWindow(window_name, WINDOW_NORMAL); //create a window
	int x = 0;
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
		Gaussian_Blur_CUDA(gray.data, gray.rows, gray.cols);

		//Image display
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
	}

	return 0;
}