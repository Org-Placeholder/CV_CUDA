#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "updownsample.h"
#include <iostream>
#include <stdio.h>

__global__ void Downsample_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);
__global__ void Upsample_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);

Mat upsample(Mat input_image)
{
    int SAMPLE_RATIO = 4;

    Mat channels[3];
    split(input_image, channels);

    int Height = input_image.rows;
    int Width = input_image.cols;

    int new_height = Height * SAMPLE_RATIO;
    int new_width = Width * SAMPLE_RATIO;

    for (int j = 0; j < 2; j++)
    {
        unsigned char* Dev_Input_Image = NULL;
        //allocate the memory in gpu
        cudaMalloc((void**)&Dev_Input_Image, Height * Width * sizeof(unsigned char));
        //copy image to gpu
        cudaMemcpy(Dev_Input_Image, channels[j].data, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

        //allocating space for output image
        unsigned char* Dev_Output_Image = NULL;
        cudaMalloc((void**)&Dev_Output_Image, new_height * new_width * sizeof(unsigned char));

        //specifying grid and block size.
        //since there doesnt need to be any inter-thread communication, we keep block size (1,1)
        dim3 Grid_Image(Height, Width);
        dim3 Block_size(1, 1);

        size_t shm_size = 4 * sizeof(unsigned long long);
        Upsample_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image);

        unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char*) * new_height * new_width);
        //copy processed data back to cpu from gpu
        cudaMemcpy(channels[j].data, Dev_Output_Image, new_height * new_width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
        if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));

        //free gpu mempry
        cudaFree(Dev_Input_Image);
        cudaFree(Dev_Output_Image);
    }



    Mat result;
    merge(channels, 3, result);
    return result;

}

Mat* downsample(Mat input_image)
{
    int SAMPLE_RATIO = 4;

    Mat channels[3];
    split(input_image, channels);

    int Height = input_image.rows;
    int Width = input_image.cols;

    if (Height % SAMPLE_RATIO == 0)
    {
        Height = input_image.rows + Height % SAMPLE_RATIO;
        resize(channels[0], channels[0], cv::Size(input_image.rows + Height % SAMPLE_RATIO, input_image.cols), 0, 0);
        resize(channels[1], channels[1], cv::Size(input_image.rows + Height % SAMPLE_RATIO, input_image.cols), 0, 0);
        resize(channels[2], channels[2], cv::Size(input_image.rows + Height % SAMPLE_RATIO, input_image.cols), 0, 0);
    }

    if (Width % SAMPLE_RATIO == 0)
    {
        Width = input_image.cols + Width % SAMPLE_RATIO;
        resize(channels[0], channels[0], cv::Size(input_image.rows, input_image.cols + Width % SAMPLE_RATIO), 0, 0);
        resize(channels[1], channels[1], cv::Size(input_image.rows, input_image.cols + Width % SAMPLE_RATIO), 0, 0);
        resize(channels[2], channels[2], cv::Size(input_image.rows, input_image.cols + Width % SAMPLE_RATIO), 0, 0);
    }

    int new_height = Height / SAMPLE_RATIO;
    int new_width = Width / SAMPLE_RATIO;

    Mat new_channels[3];

    new_channels[2].data = channels[2].data;
    new_channels[1].data = (unsigned char*)malloc(new_height * new_width * sizeof(unsigned char));
    new_channels[0].data = (unsigned char*)malloc(new_height * new_width * sizeof(unsigned char));

    for (int j = 0; j < 2; j++)
    {
        unsigned char* Dev_Input_Image = NULL;
        //allocate the memory in gpu
        cudaMalloc((void**)&Dev_Input_Image, Height * Width * sizeof(unsigned char));
        //copy image to gpu
        cudaMemcpy(Dev_Input_Image, channels[j].data, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

        //allocating space for output image
        unsigned char* Dev_Output_Image = NULL;
        cudaMalloc((void**)&Dev_Output_Image, new_height * new_width * sizeof(unsigned char));

        //specifying grid and block size.
        //since there doesnt need to be any inter-thread communication, we keep block size (1,1)
        dim3 Grid_Image(new_height, new_width);
        dim3 Block_size(1, 1);

        size_t shm_size = 4 * sizeof(unsigned long long);
        Downsample_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image);

        unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char*) * new_height * new_width);
        //copy processed data back to cpu from gpu
        cudaMemcpy(new_channels[j].data, Dev_Output_Image, new_height * new_width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
        if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));

        //free gpu mempry
        cudaFree(Dev_Input_Image);
        cudaFree(Dev_Output_Image);
    }

    return new_channels;
}

__global__ void Downsample_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{

    int SAMPLE_RATIO = 4;

    int i = blockIdx.x;
    int j = blockIdx.y;
    int i_start = i * SAMPLE_RATIO;
    int j_start = j * SAMPLE_RATIO;

    int height = gridDim.y;
    int big_height = height * SAMPLE_RATIO;

    int sum = 0;

    for (int k = 0; k < SAMPLE_RATIO; k++)
    {
        for (int l = 0; l < SAMPLE_RATIO; l++)
        {
            sum += Dev_Input_Image[((i_start + k) * big_height) + j_start + l];
        }
    }
   
    sum /= SAMPLE_RATIO;
    Dev_Output_Image[(i * height) + j] = sum;
}

__global__ void Upsample_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{

    int SAMPLE_RATIO = 4;

    int i = blockIdx.x;
    int j = blockIdx.y;
    int i_start = i * SAMPLE_RATIO;
    int j_start = j * SAMPLE_RATIO;

    int height = gridDim.y;
    int big_height = height * SAMPLE_RATIO;

    int sum = 0;

    for (int k = 0; k < SAMPLE_RATIO; k++)
    {
        for (int l = 0; l < SAMPLE_RATIO; l++)
        {
            Dev_Output_Image[((i_start + k) * big_height) + j_start + l] = Dev_Input_Image[i * height + j];
        }
    }

}