extern "C" {
#include "cuda_filters.h"
}

#define BLOCK_DIM 512
__global__ void
gray_kernel( pixel * p, int max){
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < max){
        int moy ;

        moy = (p[i].r + p[i].g + p[i].b)/3 ;
        if ( moy < 0 ) moy = 0 ;
        if ( moy > 255 ) moy = 255 ;

        p[i].r = moy ;
        p[i].g = moy ;
        p[i].b = moy ;
    }
}

int grid_dim(int data_size, int block_dim){
    return data_size/block_dim + 1;
}

void gray_filter_cuda(pixel* p, int width, int height)
{
    pixel * d_p;
    cudaMalloc((void **)&d_p, width * height * sizeof(pixel));
    cudaMemcpy(d_p, p, width * height * sizeof(pixel), cudaMemcpyHostToDevice);
    gray_kernel<<<grid_dim(width*height, BLOCK_DIM),BLOCK_DIM>>>(d_p, width * height);

    cudaMemcpy(p, d_p, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);
}
