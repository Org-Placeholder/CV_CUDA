#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Sobel_CUDA.h"
#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void Salt_Pepper_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image, curandState_t* state);
__global__ void init(curandState_t* state, int seed);

unsigned char* Salt_Pepper(unsigned char* Input_Image, int Height, int Width) {

    curandState_t* Dev_Rand_State;
    cudaMalloc((void**)&Dev_Rand_State, sizeof(curandState_t));
    init << <1, 1 >> > (Dev_Rand_State, time(0));

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
    Salt_Pepper_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image, Dev_Rand_State);

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

__global__ void Salt_Pepper_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image, curandState_t* state)
{

    int i = blockIdx.x;
    int j = blockIdx.y;

    int width = gridDim.y;
    
    Dev_Output_Image[(i)*width + (j)] = Dev_Input_Image[(i)*width + (j)];
    int x, y, z;
    x = curand(state) % 2;
    y = curand(state) % 2;
    z = curand(state) % 2;
    if (x == 1 && y == 1 && z == 1)
    {
        Dev_Output_Image[(i)*width + (j)] = 255;
    }
    else if (x == 0 && y == 0 && z == 1)
    {
        Dev_Output_Image[(i)*width + (j)] = 0;
    }
}

__global__ void init(curandState_t* state, int seed)
{
    /* we have to initialize the state */
    curand_init(0, 0, 0, state);
}