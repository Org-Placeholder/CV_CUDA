#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Sobel_CUDA.h"
#include <iostream>
#include <stdio.h>

__global__ void Sobel_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);

unsigned char* Sobel_CUDA(unsigned char* Input_Image, int Height, int Width) { 
    
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
    dim3 Grid_Image(Height - 2, Width - 2);
    dim3 Block_size(1, 1);

    size_t shm_size = 4 * sizeof(unsigned long long);
	Sobel_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image);

    unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char*) * Height * Width);
	//copy processed data back to cpu from gpu
    cudaMemcpy(Input_Image, Dev_Output_Image, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));
	

	//free gpu mempry
	cudaFree(Dev_Input_Image);
    cudaFree(Dev_Output_Image);

    return result;
}

__global__ void Sobel_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{  

    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    int horizontalDiff = 0;
    int verticalDiff = 0;

    int width = gridDim.y + 2;

    horizontalDiff = Dev_Input_Image[(i - 1) * width + (j + 1) ] - Dev_Input_Image[(i - 1) * width + (j - 1)];
    horizontalDiff += 2 * (Dev_Input_Image[(i) *width + (j + 1)] - Dev_Input_Image[(i) *width + (j - 1)]);
    horizontalDiff = Dev_Input_Image[(i + 1) * width + (j + 1)] - Dev_Input_Image[(i+1) * width + (j - 1) ];

    verticalDiff = -Dev_Input_Image[(i - 1) * width + (j - 1)] - Dev_Input_Image[(i - 1) * width + (j + 1)];
    verticalDiff += Dev_Input_Image[(i + 1) * width + (j + 1)] + Dev_Input_Image[(i + 1) * width + (j - 1)];
    verticalDiff += 2 * (Dev_Input_Image[(i+1) * width + (j)] - Dev_Input_Image[(i - 1) * width + (j)]);

    Dev_Output_Image[(i - 1) * width + (j - 1)] = sqrt((float)(horizontalDiff * horizontalDiff + verticalDiff * verticalDiff))/4;
}