#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Sobel_CUDA.h"
#include <iostream>
#include <stdio.h>

__global__ void Canny_CUDA_kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);
__global__ void Edge_Tracking_CUDA_kernel(unsigned char* Dev_Input_Image);
unsigned char* Canny_CUDA(unsigned char* Input_Image, int Height, int Width) {

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
    Canny_CUDA_kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image, Dev_Output_Image);

    

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));
    
    int max_iter = 250;
    
    while (max_iter != 0)
    {
        dim3 Grid_Image(Height, Width);
        dim3 Block_size(1, 1);

        size_t shm_size = 4 * sizeof(unsigned long long);
        Edge_Tracking_CUDA_kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Output_Image);
        cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
        if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));

        max_iter--;
    }
    unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char*) * Height * Width);
    //copy processed data back to cpu from gpu
    cudaMemcpy(Input_Image, Dev_Output_Image, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //free gpu mempry
    cudaFree(Dev_Input_Image);
    cudaFree(Dev_Output_Image);

    return result;
}

__global__ void Canny_CUDA_kernel(unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{

    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;
    int p = i - 1;
    int q = j - 1;
    int horizontalDiff = 0;
    int verticalDiff = 0;

    int width = gridDim.y + 2;

    horizontalDiff = Dev_Input_Image[(i - 1) * width + (j + 1)] - Dev_Input_Image[(i - 1) * width + (j - 1)];
    horizontalDiff += 2 * (Dev_Input_Image[(i)*width + (j + 1)] - Dev_Input_Image[(i)*width + (j - 1)]);
    horizontalDiff = Dev_Input_Image[(i + 1) * width + (j + 1)] - Dev_Input_Image[(i + 1) * width + (j - 1)];

    verticalDiff = -Dev_Input_Image[(i - 1) * width + (j - 1)] - Dev_Input_Image[(i - 1) * width + (j + 1)];
    verticalDiff += Dev_Input_Image[(i + 1) * width + (j + 1)] + Dev_Input_Image[(i + 1) * width + (j - 1)];
    verticalDiff += 2 * (Dev_Input_Image[(i + 1) * width + (j)] - Dev_Input_Image[(i - 1) * width + (j)]);

    //Dev_Output_Image[(i - 1) * width + (j - 1)] = sqrt((float)(horizontalDiff * horizontalDiff + verticalDiff * verticalDiff)) / 4;

    int x = Dev_Input_Image[(i) * (width) + j];

    double ratio = verticalDiff / horizontalDiff;
    double x1 = ratio;
    double x3 = x1 * ratio * ratio;
    double x5 = x3 * ratio * ratio;
    double x7 = x5 * ratio * ratio;
    double x9 = x7 * ratio * ratio;


    double ang = x1 - x3/3 + x5/5 - x7/7 + x9/9 ;

    ang *= 180 / 3.1415;

    if ((ang >= -22.5 && ang <= 22.5) || (ang >= -180 && ang < -157.5)) // (1) (i,j+1) & (i,j-1)
        if (x > Dev_Input_Image[(p + 1) * (width) + q] && x > Dev_Input_Image[(p + 1) * (width) + (q + 2)])
            Dev_Output_Image[i * width + j] = x;
        else
            Dev_Output_Image[i * width + j] = 0;
    else if ((ang >= 22.5 && ang <= 67.5) || (ang < -112.5 && ang >= -157.5)) // (2)  (i-1,j+1) & (i+1,j-1)
        if (x > Dev_Input_Image[p * (width) + q + 2] && x > Dev_Input_Image[(p + 2) * (width) + q])
            Dev_Output_Image[i * width + j] = x;
        else
            Dev_Output_Image[i * width + j] = 0;
    else if ((ang >= 67.5 && ang < 112.5) || (ang < -67.5 && ang >= -112.5)) // (3)  (i+1,j) & (i-1,j)
        if (x > Dev_Input_Image[(p + 2) * (width) + (q + 1)] && x > Dev_Input_Image[p * (width) + q + 1])
            Dev_Output_Image[i * width + j] = x;
        else
            Dev_Output_Image[i * width + j] = 0;
    else                                                               // (4)  (i+1,j+1) & (i-1,j-1)
        if (x > Dev_Input_Image[(p + 2) * (width) + q + 2] && x > Dev_Input_Image[p * (width) + q])
            Dev_Output_Image[i * width + j] = x;
        else
            Dev_Output_Image[i * width + j] = 0;
    double max_threshold = 0.6 * 255;
    double min_threshold = 0.3 * 255;
    
    int max_thres = max_threshold;
    int min_thres = min_threshold;
            if (Dev_Output_Image[i * width + j] > max_thres)
                Dev_Output_Image[i * width + j] = 255;
            else if(Dev_Output_Image[i * width + j] < min_thres)
                Dev_Output_Image[i * width + j] = 0;
}

__global__ void Edge_Tracking_CUDA_kernel(unsigned char* Dev_Input_Image)
{

    int i = blockIdx.x;
    int j = blockIdx.y;

    int height = gridDim.x;
    int width = gridDim.y;
    if (Dev_Input_Image[i * width + j] == 255)
    {
        for (int k = -2; k <= 2; k++)
        {
            for (int l = -2; l <= 2; l++)
            {
                int x = i + k;
                int y = j + l;
                if (x >= 0 && y >= 0 && x < height && y < width)
                {
                    if (Dev_Input_Image[x * width + y] != 0 && Dev_Input_Image[x * width + y] != 255)
                    {
                        Dev_Input_Image[x * width + y] = 255;
                        return;
                    }
                }
            }
        }
    }
    

}