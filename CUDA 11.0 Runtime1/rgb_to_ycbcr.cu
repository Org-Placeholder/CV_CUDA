#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rgb_to_ycbcr.h"
#include <iostream>
#include <stdio.h>

using namespace cv;

__global__ void rgb2ycbcr_CUDA_Kernel(unsigned char* Dev_Input_Image_r, unsigned char* Dev_Input_Image_g, unsigned char* Dev_Input_Image_b, unsigned char* Dev_Output_Image_y, unsigned char* Dev_Output_Image_cb, unsigned char* Dev_Output_Image_cr);

__global__ void ycbcr2rgb_CUDA_Kernel(unsigned char* Dev_Input_Image_y, unsigned char* Dev_Input_Image_cb, unsigned char* Dev_Input_Image_cr, unsigned char* Dev_Output_Image_r, unsigned char* Dev_Output_Image_g, unsigned char* Dev_Output_Image_b);

Mat rgb2ycbcr(Mat input_image)
{
	Mat channels[3];
	split(input_image, channels);

    int Height = input_image.rows;
    int Width = input_image.cols;

    unsigned char* Dev_Input_Image_r = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image_r, Height * Width * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image_r, channels[2].data, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsigned char* Dev_Input_Image_g = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image_g, Height * Width * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image_g, channels[1].data, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsigned char* Dev_Input_Image_b = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image_b, Height * Width * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image_b, channels[0].data, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //allocating space for output image
    unsigned char* Dev_Output_Image_y = NULL;
    cudaMalloc((void**)&Dev_Output_Image_y, Height * Width * sizeof(unsigned char));

    //allocating space for output image
    unsigned char* Dev_Output_Image_cb = NULL;
    cudaMalloc((void**)&Dev_Output_Image_cb, Height * Width * sizeof(unsigned char));

    //allocating space for output image
    unsigned char* Dev_Output_Image_cr = NULL;
    cudaMalloc((void**)&Dev_Output_Image_cr, Height * Width * sizeof(unsigned char));

    //specifying grid and block size.
    //since there doesnt need to be any inter-thread communication, we keep block size (1,1)
    dim3 Grid_Image(Height, Width);
    dim3 Block_size(1, 1);

    size_t shm_size = 4 * sizeof(unsigned long long);
    rgb2ycbcr_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image_r, Dev_Input_Image_g, Dev_Input_Image_b, Dev_Output_Image_y, Dev_Output_Image_cb, Dev_Output_Image_cr);

    //copy processed data back to cpu from gpu
    cudaMemcpy(channels[2].data, Dev_Output_Image_y, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //copy processed data back to cpu from gpu
    cudaMemcpy(channels[1].data, Dev_Output_Image_cb, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //copy processed data back to cpu from gpu
    cudaMemcpy(channels[0].data, Dev_Output_Image_cr, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));


    //free gpu mempry
    cudaFree(Dev_Input_Image_r);
    cudaFree(Dev_Input_Image_g);
    cudaFree(Dev_Input_Image_b);
    cudaFree(Dev_Output_Image_y);
    cudaFree(Dev_Output_Image_cb);
    cudaFree(Dev_Output_Image_cr);


    Mat result;
    merge(channels, 3, result);

    return result;


}

Mat ycbcr2rgb(Mat input_image)
{
    Mat channels[3];
    split(input_image, channels);

    int Height = input_image.rows;
    int Width = input_image.cols;

    unsigned char* Dev_Input_Image_y = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image_y, Height * Width * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image_y, channels[2].data, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsigned char* Dev_Input_Image_cb = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image_cb, Height * Width * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image_cb, channels[1].data, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsigned char* Dev_Input_Image_cr = NULL;
    //allocate the memory in gpu
    cudaMalloc((void**)&Dev_Input_Image_cr, Height * Width * sizeof(unsigned char));
    //copy image to gpu
    cudaMemcpy(Dev_Input_Image_cr, channels[0].data, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //allocating space for output image
    unsigned char* Dev_Output_Image_r = NULL;
    cudaMalloc((void**)&Dev_Output_Image_r, Height * Width * sizeof(unsigned char));

    //allocating space for output image
    unsigned char* Dev_Output_Image_g = NULL;
    cudaMalloc((void**)&Dev_Output_Image_g, Height * Width * sizeof(unsigned char));

    //allocating space for output image
    unsigned char* Dev_Output_Image_b = NULL;
    cudaMalloc((void**)&Dev_Output_Image_b, Height * Width * sizeof(unsigned char));

    //specifying grid and block size.
    //since there doesnt need to be any inter-thread communication, we keep block size (1,1)
    dim3 Grid_Image(Height, Width);
    dim3 Block_size(1, 1);

    size_t shm_size = 4 * sizeof(unsigned long long);
    ycbcr2rgb_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (Dev_Input_Image_y, Dev_Input_Image_cb, Dev_Input_Image_cr, Dev_Output_Image_r, Dev_Output_Image_g, Dev_Output_Image_b);

    //copy processed data back to cpu from gpu
    cudaMemcpy(channels[2].data, Dev_Output_Image_r, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //copy processed data back to cpu from gpu
    cudaMemcpy(channels[1].data, Dev_Output_Image_g, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //copy processed data back to cpu from gpu
    cudaMemcpy(channels[0].data, Dev_Output_Image_b, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));


    //free gpu mempry
    cudaFree(Dev_Input_Image_y);
    cudaFree(Dev_Input_Image_cb);
    cudaFree(Dev_Input_Image_cr);
    cudaFree(Dev_Output_Image_r);
    cudaFree(Dev_Output_Image_g);
    cudaFree(Dev_Output_Image_b);


    Mat result;
    merge(channels, 3, result);

    return result;
}

__global__ void rgb2ycbcr_CUDA_Kernel(unsigned char* Dev_Input_Image_r, unsigned char* Dev_Input_Image_g, unsigned char* Dev_Input_Image_b, unsigned char* Dev_Output_Image_y, unsigned char* Dev_Output_Image_cb, unsigned char* Dev_Output_Image_cr)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    int height = gridDim.y;

    float r = Dev_Input_Image_r[(i * height) + j];
    float g = Dev_Input_Image_g[(i * height) + j];
    float b = Dev_Input_Image_b[(i * height) + j];

    r /= 255;
    g /= 255;
    b /= 255;

    float y = r * 0.299 + g * 0.587 + b * 0.114;
    float cb = r * (-0.168736) + g * (-0.331264) + b * 0.5;
    float cr = r * 0.500 + g * (-0.418688) + b * (-0.081312);

    Dev_Output_Image_y[(i * height) + j] = (unsigned char)(y*219 + 16);
    Dev_Output_Image_cb[(i * height) + j] = (unsigned char)(cb*224 + 128);
    Dev_Output_Image_cr[(i * height) + j] = (unsigned char)(cr*224 + 128);

}

__global__ void ycbcr2rgb_CUDA_Kernel(unsigned char* Dev_Input_Image_y, unsigned char* Dev_Input_Image_cb, unsigned char* Dev_Input_Image_cr, unsigned char* Dev_Output_Image_r, unsigned char* Dev_Output_Image_g, unsigned char* Dev_Output_Image_b)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    int height = gridDim.y;
    
    float y = (Dev_Input_Image_y[(i * height) + j] - 16);
    y /= 219;
    float cb = (Dev_Input_Image_cb[(i * height) + j] - 128);
    cb /= 224;
    float cr = (Dev_Input_Image_cr[(i * height) + j] - 128);
    cr /= 224;

    float r = y * 1.0 + cb * 0 + cr * 1.402;
    float g = y * 1.0 + cb * (-0.344136) + cr * (-0.714136);
    float b = y * 1.0 + cb * (1.772) + cr * 0;

    //printf("%d", Dev_Input_Image_y[(i * height) + j]);

    r *= 255;
    g *= 255;
    b *= 255;

    Dev_Output_Image_r[(i * height) + j] = (unsigned char)r;
    Dev_Output_Image_g[(i * height) + j] = (unsigned char)g;
    Dev_Output_Image_b[(i * height) + j] = (unsigned char)b;
}