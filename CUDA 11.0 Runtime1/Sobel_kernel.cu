#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Sobel_CUDA.h"
#include <iostream>
#include <stdio.h>

__global__ void Sobel_CUDA_Kernel(int* horizontal, int* vertical, unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image);

unsigned char* Sobel_CUDA(unsigned char* Input_Image, int Height, int Width, int channels) {
	   
    
    unsigned char* Dev_Input_Image = NULL;

	//allocate the memory in gpu
	cudaMalloc((void**)&Dev_Input_Image, Height * Width * sizeof(unsigned char));
	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsigned char* Dev_Output_Image = NULL;
    cudaMalloc((void**)&Dev_Output_Image, Height * Width * sizeof(unsigned char));

    //int horizontal[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int* horizontal = (int*)malloc(sizeof(int) * 9);
    horizontal[0] = -1;
    horizontal[1] = 0;
    horizontal[2] = 1;
    horizontal[3] = -2;
    horizontal[4] = 0;
    horizontal[5] = 2;
    horizontal[6] = -1;
    horizontal[7] = 0;
    horizontal[8] = 1;

    int* horizontal_CUDA = NULL;
    cudaMalloc((void**)&horizontal_CUDA, sizeof(horizontal));
    cudaMemcpy(horizontal_CUDA, horizontal, 9 * sizeof(int), cudaMemcpyHostToDevice);

    //int vertical[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };
    int* vertical = (int*)malloc(sizeof(int) * 9);
    vertical[0] = -1;
    vertical[1] = 0;
    vertical[2] = 1;
    vertical[3] = -2;
    vertical[4] = 0;
    vertical[5] = 2;
    vertical[6] = -1;
    vertical[7] = 0;
    vertical[8] = 1;
    int* vertical_CUDA = NULL;
    cudaMalloc((void**)&vertical_CUDA, sizeof(vertical));
    cudaMemcpy(vertical_CUDA, vertical, 9 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid_Image(Height - 2, Width - 2);
    dim3 Block_size(1, 1);
    //std::cout << horizontal[0];
    size_t shm_size = 4 * sizeof(unsigned long long);
	Sobel_CUDA_Kernel << <Grid_Image, Block_size, shm_size >> > (horizontal_CUDA, vertical_CUDA, Dev_Input_Image, Dev_Output_Image);
    //std::cout << "REACHED789";
    unsigned char* result = (unsigned char*)malloc(sizeof(unsigned char*) * Height * Width);
	//copy processed data back to cpu from gpu
    cudaMemcpy(Input_Image, Dev_Output_Image, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if (cudaerror != cudaSuccess) fprintf(stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName(cudaerror));
	//cudaMemcpy(Input_Image, Dev_Output_Image, Height * Width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	//free gpu mempry
	cudaFree(Dev_Input_Image);
    cudaFree(vertical_CUDA);
    cudaFree(horizontal_CUDA);
    cudaFree(Dev_Output_Image);

    return result;
}

__global__ void Sobel_CUDA_Kernel(int* horizontal, int* vertical, unsigned char* Dev_Input_Image, unsigned char* Dev_Output_Image)
{  

    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;

    int horizontalDiff = 0;
    int verticalDiff = 0;
    
    horizontalDiff = Dev_Input_Image[(i - 1) * 640 + (j + 1) ] - Dev_Input_Image[(i - 1) * 640 + (j - 1)];
    horizontalDiff += 2 * (Dev_Input_Image[(i) * 640 + (j + 1)] - Dev_Input_Image[(i) * 640 + (j - 1)]);
    horizontalDiff = Dev_Input_Image[(i + 1) * 640 + (j + 1)] - Dev_Input_Image[(i+1) * 640 + (j - 1) ];

    verticalDiff = Dev_Input_Image[(i + 1) * 640 + (j - 1)] - Dev_Input_Image[(i + 1) * 640 + (j + 1)];
    verticalDiff += 2 * (Dev_Input_Image[(i) * 640 + (j - 1)] - Dev_Input_Image[(i) * 640 + (j + 1)]);
    verticalDiff = Dev_Input_Image[(i - 1) * 640 + (j - 1)] - Dev_Input_Image[(i-1) * 640 + (j + 1)];

    Dev_Output_Image[(i - 1) * 640 + (j - 1)] = sqrt((float)(horizontalDiff * horizontalDiff + verticalDiff * verticalDiff));

    //printf("%d", horizontal[0]);
}