#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Sobel_CUDA.h"
#include <iostream>
#include <stdio.h>

__global__ void Gaussian_Blur_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);

unsigned char* Gaussian_Blur_CUDA(unsigned char* Input_Image, int Height, int Width) {

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
    dim3 Grid_Image(Height , Width );
    dim3 Block_size(1, 1);
    
    size_t shm_size = 4 * sizeof(unsigned long long);
    Gaussian_Blur_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image);

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

__global__ void Gaussian_Blur_CUDA_Kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    int height = gridDim.x;
    int width = gridDim.y;

    float val = 0;
  
    for (int k = -1; k <= 1; k++)
    {
        for (int l = -1; l <= 1; l++)
        {
            int x = i + k;
            int y = j + l;
            int z = abs(l) + abs(k);
            printf(l + "Hello" + k);
            if (x >= 0 && y >= 0 && x < height && y < width)
            {
                if (z == 1)
                {
                    val += 24 * Dev_Input_Image[x * width + y];
                }
                else if (z == 2)
                {
                    if (l == 0 || k ==0 )
                    {
                        val += 6 * Dev_Input_Image[x * width + y];
                    }
                    else
                    {
                        val += 16 * Dev_Input_Image[x * width + y];
                    }
                }
                else if (z == 3)
                {
                    val += 4 *  Dev_Input_Image[x * width + y];
                }
                else if (z == 4)
                {
                    val += Dev_Input_Image[x * width + y];
                }
                else
                {
                    val += 36 * Dev_Input_Image[x * width + y];
                }
            }
        }
    }

    val /= 256;

    Dev_Output_Image[(i * width) + j] = val;
}