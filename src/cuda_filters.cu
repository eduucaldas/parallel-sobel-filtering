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

int grid_dim(int image_size, int block_dim){
    return image_size/block_dim + 1;
}

void gray_filter_cuda(pixel* p, int width, int height)
{
    int size_image = width*height;
    pixel * d_p;
    cudaMalloc((void **)&d_p, size_image * sizeof(pixel));
    cudaMemcpy(d_p, p, size_image * sizeof(pixel), cudaMemcpyHostToDevice);
    gray_kernel<<<grid_dim(size_image, BLOCK_DIM),BLOCK_DIM>>>(d_p, size_image);

    cudaMemcpy(p, d_p, size_image * sizeof(pixel), cudaMemcpyDeviceToHost);
}

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

__device__ int end;

__global__ void
blur_kernel(pixel* p, pixel* p_0, int width, int height, int size) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    int j, k;
    j = i / width;
    k = i % width;

    if (i < width * height && k >= size && k < width - size
        && ( (j >= size && j < height / 10 - size)
        || (j >= (height * 9) / 10 + size && j < height - size) )){
        int stencil_j, stencil_k;
        int t_r = 0;
        int t_g = 0;
        int t_b = 0;

        for (stencil_j = -size; stencil_j <= size; stencil_j++) {
            for (stencil_k = -size; stencil_k <= size; stencil_k++) {
                t_r += p_0[CONV(j+stencil_j,k+stencil_k,width)].r ;
                t_g += p_0[CONV(j+stencil_j,k+stencil_k,width)].g ;
                t_b += p_0[CONV(j+stencil_j,k+stencil_k,width)].b ;
            }
        }

        p[i].r = t_r / ( (2*size+1)*(2*size+1) ) ;
        p[i].g = t_g / ( (2*size+1)*(2*size+1) ) ;
        p[i].b = t_b / ( (2*size+1)*(2*size+1) ) ;
    }
}

__global__ void
blur_diff(pixel* p, pixel* p_0, int width, int height, int threshold) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    int j, k;
    j = i / width;
    k = i % width;

    if (i < width * height && 
        j > 0 && j < height - 1 && k > 0 && k < width - 1){
        float diff_r;
        float diff_g;
        float diff_b;

        diff_r = p[i].r - p_0[i].r;
        diff_g = p[i].g - p_0[i].g;
        diff_b = p[i].b - p_0[i].b;

        if (diff_r > threshold || -diff_r > threshold
            || diff_g > threshold || -diff_g > threshold
            || diff_b > threshold || -diff_b > threshold) {
            atomicExch(&end, 0);
        }

        p_0[i].r = p[i].r;
        p_0[i].g = p[i].g;
        p_0[i].b = p[i].b;
    }
}

void blur_filter_cuda(pixel* p, int width, int height, int size, int threshold) {
    pixel *d_p, *p_0;

    int end_host;
    int n_iter = 0;

    cudaMalloc((void **)&d_p, width * height * sizeof(pixel));
    
    cudaMalloc((void **)&p_0, width * height * sizeof(pixel));
    cudaMemcpy(p_0, p, width * height * sizeof(pixel), cudaMemcpyHostToDevice);
    
    do {
        end_host = 1;
        n_iter++;

        cudaMemcpyToSymbol(end, &end_host, sizeof(int), 0, cudaMemcpyHostToDevice);

        cudaMemcpy(d_p, p_0, width * height * sizeof(pixel), cudaMemcpyDeviceToDevice);

        blur_kernel<<<grid_dim(width*height, BLOCK_DIM),BLOCK_DIM>>>(d_p, p_0, width, height, size);
        cudaDeviceSynchronize();
        blur_diff<<<grid_dim(width*height, BLOCK_DIM),BLOCK_DIM>>>(d_p, p_0, width, height, threshold);
        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&end_host, end, sizeof(int), 0, cudaMemcpyDeviceToHost);

    } while(threshold > 0 && !end_host);

    cudaMemcpy(p, p_0, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);
}
