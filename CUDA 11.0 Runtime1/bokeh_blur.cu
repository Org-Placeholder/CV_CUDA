/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Sobel_CUDA.h"
#include <iostream>
#include <stdio.h>*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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
using namespace cv;
using namespace std;

__global__ void Bokeh_Blur_CUDA_Kernel(unsigned char* Dev_Input_Image, float* Dev_Output_Image_2 , unsigned char* image,  int h,  int w);
__global__ void Bokeh_Blur_Cast_Kernel(unsigned char* Dev_Output_Image, float* Dev_Output_Image_2, int h);
unsigned char* Bokeh_Blur_CUDA(unsigned char* Input_Image , int Height, int Width , unsigned char* Image, int h, int w) {

    unsigned char* Dev_Input_Image = NULL;
    unsigned char* image = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image, Height * Width * sizeof(unsigned char));
    cudaMalloc((void**)&image, h * w * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(image, Image, h * w * sizeof(unsigned char), cudaMemcpyHostToDevice);
    //allocating space for output image
    unsigned char* Dev_Output_Image = NULL;
    cudaMalloc((void**)&Dev_Output_Image, Height * Width * sizeof(unsigned char));
    
    float* Dev_Output_Image_2 = NULL;
    cudaMalloc((void**)&Dev_Output_Image_2, Height * Width * sizeof(float));

    //specifying grid and block size.
    //since there doesnt need to be any inter-thread communication, we keep block size (1,1)
    dim3 Grid_Image(Height, Width);
    dim3 Block_size(h/4, w/4);

    size_t shm_size = 4 * sizeof(unsigned long long);
    //Bokeh_Blur_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image);
    Bokeh_Blur_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image_2 , image, h, w);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize at kernel 1: %s\n", cudaGetErrorName(cudaerror));
    Bokeh_Blur_Cast_Kernel<< <Grid_Image , (1,1) , shm_size >> > (Dev_Output_Image, Dev_Output_Image_2, h);
    cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize at kernel 2: %s\n", cudaGetErrorName(cudaerror));
    unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char*) * Height * Width);
    //copy processed data back to cpu from gpu
    cudaMemcpy(Input_Image, Dev_Output_Image, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));
    //free gpu mempry
    cudaFree(Dev_Input_Image);
    cudaFree(Dev_Output_Image);
    cudaFree(image);

    return result;
}

__global__ void Bokeh_Blur_CUDA_Kernel(unsigned char* Dev_Input_Image, float* Dev_Output_Image_2 , unsigned char* image,  int h,  int w)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    
    int height = gridDim.x;
    int width = gridDim.y;
    float val = 0;
    float total = 0;
    int k_start = threadIdx.x;
    int l_start = threadIdx.y;
    k_start *= 4;
    l_start *= 4;
    for (int p = 0; p < 4; p++)
    {
        for (int q = 0; q < 4; q++)
        {
            int k = k_start + p;
            int l = l_start + q;
            int x = i + k;
            int y = j + l;
            if (x >= 0 && y >= 0 && x < height && y < width)
            {
                    val += Dev_Input_Image[x * width + y] * image[k * h + l];
                    total += image[k * h + l];
             
            }
        }
    }

    //val *= 30;
    //printf("val = %f, total = %f", val, total);
    atomicAdd(&Dev_Output_Image_2[(i * width) + j], val);
    //Dev_Output_Image_2[(i * width) + j] += val;
    //__threadfence();

}
__global__ void Bokeh_Blur_Cast_Kernel(unsigned char* Dev_Output_Image, float* Dev_Output_Image_2, int h)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int width = gridDim.y;
    Dev_Output_Image_2[(i * width) + j] /= 3.14*h*255*h/4;
    if (Dev_Output_Image_2[(i * width) + j] > 255)
    {
        Dev_Output_Image_2[(i * width) + j] = 255;
    }
    if (Dev_Output_Image_2[(i * width) + j] < 0)
    {
        Dev_Output_Image_2[(i * width) + j] = 0;
    }
    
    Dev_Output_Image[(i * width) + j] = Dev_Output_Image_2[(i * width) + j];

}


