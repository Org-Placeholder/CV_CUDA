#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Gaussian_Blur_Seperated.h"
#include <iostream>
#include <stdio.h>

__global__ void Gaussian_Blur_Vertical_kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);
__global__ void Gaussian_Blur_Horizontal_kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);

unsigned char* Gaussian_Blur_Seperated(unsigned char* Input_Image, int Height, int Width) 
{

    unsigned char* Dev_Input_Image = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image, Height * Width * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //allocating space for output image
    unsigned char* Dev_Output_Image = NULL;
    cudaMalloc((void**)&Dev_Output_Image, Height * Width * sizeof(unsigned char));

    //specifying grid and block size.
    //since there doesnt need to be any inter-thread communication, we keep block size (1,1)
    dim3 Grid_Image(Height, Width);
    dim3 Block_size(1, 1);

    size_t shm_size = 4 * sizeof(unsigned long long);
    Gaussian_Blur_Vertical_kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image);
    Gaussian_Blur_Horizontal_kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Output_Image, Dev_Input_Image);

    unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char*) * Height * Width);
    //copy processed data back to cpu from gpu
    cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));


    //free gpu mempry
    cudaFree(Dev_Input_Image);
    cudaFree(Dev_Output_Image);

    return result;
}

__global__ void Gaussian_Blur_Vertical_kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{

    int i = blockIdx.x;
    int j = blockIdx.y;

    int height = gridDim.x;
    int width = gridDim.y;

    float val = 0;
    float count = 0;
    float filter[7] = { 16.0, 13.5, 8.20, 3.56, 1.11, 0.25, 0.04 };
    for (int k = -6; k <= 6; k++)
    {
        
        int y = j + k;
        int z = abs(k);
        if (y >= 0 && y < width)
        {
            val += Dev_Input_Image[i * width + y]* filter[z];
            count += filter[z];
        }
        
    }

    val /= count;

    Dev_Output_Image[(i * width) + j] = val;
}

__global__ void Gaussian_Blur_Horizontal_kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{

    int i = blockIdx.x;
    int j = blockIdx.y;

    int height = gridDim.x;
    int width = gridDim.y;

    float val = 0;
    float count = 0;
    float filter[7] = { 16.0, 13.5, 8.20, 3.56, 1.11, 0.25, 0.04 };
    for (int k = -6; k <= 6; k++)
    {

        int x = i + k;
        int z = abs(k);
        if (x >= 0 && x < height)
        {
            val += Dev_Input_Image[x * width + j] * filter[z];
            count += filter[z];
        }

    }

    val /= count;

    Dev_Output_Image[(i * width) + j] = val;
}