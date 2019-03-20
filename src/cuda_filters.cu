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
    pixel * d_p;
    cudaMalloc((void **)&d_p, width * height * sizeof(pixel));
    cudaMemcpy(d_p, p, width * height * sizeof(pixel), cudaMemcpyHostToDevice);
    gray_kernel<<<grid_dim(width * height, BLOCK_DIM),BLOCK_DIM>>>(d_p, width * height);

    cudaMemcpy(p, d_p, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);
    cudaFree(d_p);
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
        blur_diff<<<grid_dim(width*height, BLOCK_DIM),BLOCK_DIM>>>(d_p, p_0, width, height, threshold);

        cudaMemcpyFromSymbol(&end_host, end, sizeof(int), 0, cudaMemcpyDeviceToHost);

    } while(threshold > 0 && !end_host);

    cudaMemcpy(p, p_0, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);
    cudaFree(p_0);
    cudaFree(d_p);
}



__global__ void
sobel_kernel(pixel* sobel, pixel* p_0, int width, int height) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    int j, k;
    j = i / width;
    k = i % width;

    if (j >= 1 && j < height - 1 && k >= 1 && k < width - 1){
        int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
        int pixel_blue_so, pixel_blue_s, pixel_blue_se;
        int pixel_blue_o , pixel_blue  , pixel_blue_e ;

        float deltaX_blue ;
        float deltaY_blue ;
        float val_blue;

        pixel_blue_no = p_0[CONV(j-1,k-1,width)].b ;
        pixel_blue_n  = p_0[CONV(j-1,k  ,width)].b ;
        pixel_blue_ne = p_0[CONV(j-1,k+1,width)].b ;
        pixel_blue_so = p_0[CONV(j+1,k-1,width)].b ;
        pixel_blue_s  = p_0[CONV(j+1,k  ,width)].b ;
        pixel_blue_se = p_0[CONV(j+1,k+1,width)].b ;
        pixel_blue_o  = p_0[CONV(j  ,k-1,width)].b ;
        pixel_blue    = p_0[CONV(j  ,k  ,width)].b ;
        pixel_blue_e  = p_0[CONV(j  ,k+1,width)].b ;

        deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;

        deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

        val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;


        if ( val_blue > 50 )
        {
            sobel[i].r = 255 ;
            sobel[i].g = 255 ;
            sobel[i].b = 255 ;
        } else
        {
            sobel[i].r = 0 ;
            sobel[i].g = 0 ;
            sobel[i].b = 0 ;
        }
    }
    else if(j == 0 || j == height - 1 || k == 0 || k == width - 1) {
        sobel[i].r = p_0[i].r;
        sobel[i].g = p_0[i].g;
        sobel[i].b = p_0[i].b;
    }
}

void sobel_filter_cuda(pixel* p, int width, int height) {
    pixel *sobel, *d_p;

    cudaMalloc((void **)&d_p, width * height * sizeof(pixel));
    cudaMemcpy(d_p, p, width * height * sizeof(pixel), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&sobel, width * height * sizeof(pixel));

    sobel_kernel<<<grid_dim(width*height, BLOCK_DIM),BLOCK_DIM>>>(sobel, d_p, width, height);

    cudaMemcpy(p, sobel, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);
    cudaFree(d_p);
    cudaFree(sobel);
}

