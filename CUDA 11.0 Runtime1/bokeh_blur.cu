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

__global__ void Bokeh_Blur_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image , unsigned char* image);
unsigned char* Bokeh_Blur_CUDA(unsigned char* Input_Image , int Height, int Width , unsigned char* Image) {

    unsigned char* Dev_Input_Image = NULL;
    unsigned char* image = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image, Height * Width * sizeof(unsigned char));
    cudaMalloc((void**)& Image, Height * Width * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(image, Image, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //allocating space for output image
    unsigned char* Dev_Output_Image = NULL;
    cudaMalloc((void**)&Dev_Output_Image, Height * Width * sizeof(unsigned char));

    //specifying grid and block size.
    //since there doesnt need to be any inter-thread communication, we keep block size (1,1)
    dim3 Grid_Image(Height, Width);
    dim3 Block_size(1, 1);

    size_t shm_size = 4 * sizeof(unsigned long long);
    //Bokeh_Blur_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image);
    Bokeh_Blur_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image , image);

    unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char*) * Height * Width);
    //copy processed data back to cpu from gpu
    cudaMemcpy(Input_Image, Dev_Output_Image, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));


    //free gpu mempry
    cudaFree(Dev_Input_Image);
    cudaFree(Dev_Output_Image);
    cudaFree(image);

    return result;
}

__global__ void Bokeh_Blur_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image , unsigned char* image)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    
    int height = gridDim.x;
    int width = gridDim.y;
    float val = 0;
    for (int k = 2; k < 600; k++)
    {
        val = Dev_Input_Image[k - 1] * image[k - 1];
    }
    //Dev_Output_Image[(i * width) + j] = val;

}

