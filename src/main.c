/*
 * INF560
 *
 * Image Filtering Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

#include <gif_io.h>
#include <omp.h>
#include <mpi.h>
#include <cuda_filters.h>
//------------------------ END OF FILE TREATING -------------------------------

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

// Configuration Variables
#define BLUR_SIZE 5
#define BLUR_THRESHOLD 20

MPI_Datatype MPI_PIXEL;
int root_in_world = 0;

int eq_pixel(pixel a, pixel b){
    return (a.r == b.r) && (a.g == b.g) && (a.b == b.b);
}

int black_pixel(pixel a){
    if((a.r == 255) && (a.g == 255) && (a.b == 255)) return 1;
    else if((a.r == 0) && (a.g == 0) && (a.b == 0)) return 0;
    else return 999999;
}

//------------------------ BEGIN OF FILTERS -------------------------------
// Gray Filter ------------------------------------------------------------

void gray_filter_seq(pixel* p, int width, int height){
    int j;
    for ( j = 0 ; j < width * height ; j++ )
    {
        int moy ;

        // moy = p[i][j].r/4 + ( p[i][j].g * 3/4 ) ;
        moy = (p[j].r + p[j].g + p[j].b)/3 ;
        if ( moy < 0 ) moy = 0 ;
        if ( moy > 255 ) moy = 255 ;

        p[j].r = moy ;
        p[j].g = moy ;
        p[j].b = moy ;
    }
}

void gray_filter_omp(pixel* p, int width, int height){
#pragma omp parallel
    {
        int j;

#pragma omp for schedule(static)
        for (j = 0; j < width * height; j++)
        {
            int moy ;

            // moy = p[i][j].r/4 + ( p[i][j].g * 3/4 ) ;
            moy = (p[j].r + p[j].g + p[j].b)/3 ;
            if ( moy < 0 ) moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            p[j].r = moy ;
            p[j].g = moy ;
            p[j].b = moy ;
        }
    }
}

// Gray-filter for a image splitted using MPI
// Not yet used or tested
void gray_filter_mpi(pixel* p, int width, int height){
    int rank_in_world, size_in_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_in_world);
    MPI_Comm_size(MPI_COMM_WORLD, &size_in_world);

    if(size_in_world == 1) {
        gray_filter_seq(p, width, height);
        return;
    }

    int n_slaves = size_in_world - 1;
    int chunk_size = (width * height) / n_slaves;
    int s_id, p_size, p_start;

    if(rank_in_world == root_in_world) {
        for(s_id = 1; s_id < size_in_world; s_id++) {
            p_size = chunk_size + (s_id <= (width * height) % n_slaves ? 1 : 0);
            if(s_id <= (width * height) % n_slaves) {
                p_size = chunk_size + 1;
                p_start = (s_id - 1) * (chunk_size + 1);
            }
            else {
                p_size = chunk_size;
                p_start = (s_id - 1) * chunk_size + ((width * height) % n_slaves);
            }
            MPI_Send(&p[p_start], p_size, MPI_PIXEL, s_id, s_id, MPI_COMM_WORLD);
            MPI_Recv(&p[p_start], p_size, MPI_PIXEL, s_id, s_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else {
        int j;
        pixel * p_local;

        if(rank_in_world <= (width * height) % n_slaves) {
            p_size = chunk_size + 1;
            p_start = (rank_in_world - 1) * (chunk_size + 1);
        }
        else {
            p_size = chunk_size;
            p_start = (rank_in_world - 1) * chunk_size + ((width * height) % n_slaves);
        }

        p_local = malloc(p_size * sizeof(pixel));

        MPI_Recv(p_local, p_size, MPI_PIXEL, root_in_world, rank_in_world, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (j = 0; j < width * height ; j++ )
        {
            int moy ;

            // moy = p[i][j].r/4 + ( p[i][j].g * 3/4 ) ;
            moy = (p[j].r + p[j].g + p[j].b)/3 ;
            if ( moy < 0 ) moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            p[j].r = moy ;
            p[j].g = moy ;
            p[j].b = moy ;
        }

        MPI_Send(p_local, p_size, MPI_PIXEL, root_in_world, rank_in_world, MPI_COMM_WORLD);
        free(p_local);
    }
}

// Blur Filter ------------------------------------------------------------

void blur_filter_seq(pixel * p, int width, int height, int size, int threshold){
    int n_iter = 0 ;
    int end = 1;

    /* Allocate array of new pixels */
    pixel * new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

    /* Perform at least one blur iteration */
    int j, k;
    do
    {
        /* Copy the middle part of the image */
        for(j=0; j<height; j++)
        {
            for(k=0; k<width; k++)
            {
                new[CONV(j,k,width)].r = p[CONV(j,k,width)].r ;
                new[CONV(j,k,width)].g = p[CONV(j,k,width)].g ;
                new[CONV(j,k,width)].b = p[CONV(j,k,width)].b ;
            }
        }

        end = 1 ;
        n_iter++ ;

        /* Apply blur on top part of image (10%) */
        for(j=size; j<height/10-size; j++)
        {
            for(k=size; k<width-size; k++)
            {
                int stencil_j, stencil_k ;
                int t_r = 0 ;
                int t_g = 0 ;
                int t_b = 0 ;

                for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                {
                    for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                    {
                        t_r += p[CONV(j+stencil_j,k+stencil_k,width)].r ;
                        t_g += p[CONV(j+stencil_j,k+stencil_k,width)].g ;
                        t_b += p[CONV(j+stencil_j,k+stencil_k,width)].b ;
                    }
                }

                new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
            }
        }

        /* Apply blur on the bottom part of the image (10%) */
        for(j=height*0.9+size; j<height-size; j++)
        {
            for(k=size; k<width-size; k++)
            {
                int stencil_j, stencil_k ;
                int t_r = 0 ;
                int t_g = 0 ;
                int t_b = 0 ;

                for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                {
                    for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                    {
                        t_r += p[CONV(j+stencil_j,k+stencil_k,width)].r ;
                        t_g += p[CONV(j+stencil_j,k+stencil_k,width)].g ;
                        t_b += p[CONV(j+stencil_j,k+stencil_k,width)].b ;
                    }
                }

                new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
            }
        }

        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {

                float diff_r ;
                float diff_g ;
                float diff_b ;

                diff_r = (new[CONV(j  ,k  ,width)].r - p[CONV(j  ,k  ,width)].r) ;
                diff_g = (new[CONV(j  ,k  ,width)].g - p[CONV(j  ,k  ,width)].g) ;
                diff_b = (new[CONV(j  ,k  ,width)].b - p[CONV(j  ,k  ,width)].b) ;

                if ( diff_r > threshold || -diff_r > threshold
                        ||
                        diff_g > threshold || -diff_g > threshold
                        ||
                        diff_b > threshold || -diff_b > threshold
                   ) {
                    end = 0 ;
                }

                p[CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
                p[CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
                p[CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;
            }
        }

    }
    while ( threshold > 0 && !end ) ;

    // printf( "Nb iter for image %d\n", n_iter ) ;

    free (new) ;

}

void blur_filter_omp(pixel * p, int width, int height, int size, int threshold){
    int n_iter = 0 ;
    int end = 1;

    /* Allocate array of new pixels */
    pixel * new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

    /* Perform at least one blur iteration */
    do
    {
        end = 1 ;
        n_iter++ ;

#pragma omp parallel
        {
            int j, k;

            /* Copy the middle part of the image */
#pragma omp for schedule(static)
            for(j=0; j<height; j++)
            {
                for(k=0; k<width; k++)
                {
                    new[CONV(j,k,width)].r = p[CONV(j,k,width)].r ;
                    new[CONV(j,k,width)].g = p[CONV(j,k,width)].g ;
                    new[CONV(j,k,width)].b = p[CONV(j,k,width)].b ;
                }
            }

            /* Apply blur on top part of image (10%) */
#pragma omp for schedule(static)
            for(j=size; j<height/10-size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    int stencil_j, stencil_k ;
                    int t_r = 0 ;
                    int t_g = 0 ;
                    int t_b = 0 ;

                    for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            t_r += p[CONV(j+stencil_j,k+stencil_k,width)].r ;
                            t_g += p[CONV(j+stencil_j,k+stencil_k,width)].g ;
                            t_b += p[CONV(j+stencil_j,k+stencil_k,width)].b ;
                        }
                    }

                    new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
                }
            }

            int j_end = height*0.9+size;

            /* Apply blur on the bottom part of the image (10%) */
#pragma omp for schedule(static)
            for(j=height*0.9+size; j<height-size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    int stencil_j, stencil_k ;
                    int t_r = 0 ;
                    int t_g = 0 ;
                    int t_b = 0 ;

                    for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            t_r += p[CONV(j+stencil_j,k+stencil_k,width)].r ;
                            t_g += p[CONV(j+stencil_j,k+stencil_k,width)].g ;
                            t_b += p[CONV(j+stencil_j,k+stencil_k,width)].b ;
                        }
                    }

                    new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
                }
            }

#pragma omp for schedule(static)
            for(j=1; j<height-1; j++)
            {
                for(k=1; k<width-1; k++)
                {

                    float diff_r ;
                    float diff_g ;
                    float diff_b ;

                    diff_r = (new[CONV(j  ,k  ,width)].r - p[CONV(j  ,k  ,width)].r) ;
                    diff_g = (new[CONV(j  ,k  ,width)].g - p[CONV(j  ,k  ,width)].g) ;
                    diff_b = (new[CONV(j  ,k  ,width)].b - p[CONV(j  ,k  ,width)].b) ;

                    if ( diff_r > threshold || -diff_r > threshold
                            ||
                            diff_g > threshold || -diff_g > threshold
                            ||
                            diff_b > threshold || -diff_b > threshold
                       ) {
                        end = 0 ;
                    }

                    p[CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
                    p[CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
                    p[CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;
                }
            }
        }
    }
    while ( threshold > 0 && !end ) ;

    // printf( "Nb iter for image %d\n", n_iter ) ;

    free (new) ;

}
// Sobel Filter -----------------------------------------------------
void sobel_on_pixel(pixel *p, pixel *sobel, int j, int k, int width, int totalWidth) {
    int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
    int pixel_blue_so, pixel_blue_s, pixel_blue_se;
    int pixel_blue_o , pixel_blue  , pixel_blue_e ;

    float deltaX_blue ;
    float deltaY_blue ;
    float val_blue;

    pixel_blue_n  = p[CONV(j-1,k  ,totalWidth)].b ;
    pixel_blue_ne = p[CONV(j-1,k+1,totalWidth)].b ;
    pixel_blue_so = p[CONV(j+1,k-1,totalWidth)].b ;
    pixel_blue_no = p[CONV(j-1,k-1,totalWidth)].b ;
    pixel_blue_s  = p[CONV(j+1,k  ,totalWidth)].b ;
    pixel_blue_se = p[CONV(j+1,k+1,totalWidth)].b ;
    pixel_blue_o  = p[CONV(j  ,k-1,totalWidth)].b ;
    pixel_blue    = p[CONV(j  ,k  ,totalWidth)].b ;
    pixel_blue_e  = p[CONV(j  ,k+1,totalWidth)].b ;

    deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;

    deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

    val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;


    if ( val_blue > 50 )
    {
        sobel[CONV(j  ,k  ,width)].g = 255 ;
        sobel[CONV(j  ,k  ,width)].b = 255 ;
        sobel[CONV(j  ,k  ,width)].r = 255 ;
    } else
    {
        sobel[CONV(j  ,k  ,width)].r = 0 ;
        sobel[CONV(j  ,k  ,width)].g = 0 ;
        sobel[CONV(j  ,k  ,width)].b = 0 ;
    }
}

void sobel_filter_seq(pixel* p, int width, int height){
    pixel * sobel ;

    sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;
    int j, k;
    for(j=1; j<height-1; j++)
    {
        for(k=1; k<width-1; k++)
        {
            int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
            int pixel_blue_so, pixel_blue_s, pixel_blue_se;
            int pixel_blue_o , pixel_blue  , pixel_blue_e ;

            float deltaX_blue ;
            float deltaY_blue ;
            float val_blue;

            pixel_blue_no = p[CONV(j-1,k-1,width)].b ;
            pixel_blue_n  = p[CONV(j-1,k  ,width)].b ;
            pixel_blue_ne = p[CONV(j-1,k+1,width)].b ;
            pixel_blue_so = p[CONV(j+1,k-1,width)].b ;
            pixel_blue_s  = p[CONV(j+1,k  ,width)].b ;
            pixel_blue_se = p[CONV(j+1,k+1,width)].b ;
            pixel_blue_o  = p[CONV(j  ,k-1,width)].b ;
            pixel_blue    = p[CONV(j  ,k  ,width)].b ;
            pixel_blue_e  = p[CONV(j  ,k+1,width)].b ;

            deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;

            deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

            val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;


            if ( val_blue > 50 )
            {
                sobel[CONV(j  ,k  ,width)].r = 255 ;
                sobel[CONV(j  ,k  ,width)].g = 255 ;
                sobel[CONV(j  ,k  ,width)].b = 255 ;
            } else
            {
                sobel[CONV(j  ,k  ,width)].r = 0 ;
                sobel[CONV(j  ,k  ,width)].g = 0 ;
                sobel[CONV(j  ,k  ,width)].b = 0 ;
            }
        }
    }

    for(j=1; j<height-1; j++)
    {
        for(k=1; k<width-1; k++)
        {
            p[CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
            p[CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
            p[CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
        }
    }

    free (sobel) ;

}

void sobel_filter_mini(pixel* p, int width, int height, int totalWidth){
    pixel * sobel ;

    sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

    int j, k;

    for(j=1; j<height-1; j++) {
        for(k=1; k<width-1; k++) {
            sobel_on_pixel(p, sobel, j, k, width, totalWidth);
        }
    }

    for(j=1; j<height-1; j++)
    {
        for(k=1; k<width-1; k++)
        {
            p[CONV(j  ,k  ,totalWidth)].r = sobel[CONV(j  ,k  ,width)].r ;
            p[CONV(j  ,k  ,totalWidth)].g = sobel[CONV(j  ,k  ,width)].g ;
            p[CONV(j  ,k  ,totalWidth)].b = sobel[CONV(j  ,k  ,width)].b ;
        }
    }

    free (sobel) ;

}

void sobel_filter_omp(pixel* p, int width, int height) {
    int m = height / 10, n = width / 10;

    pixel * sobel;

    sobel = (pixel*) malloc(width * height * sizeof(pixel));

#pragma omp parallel shared(p, sobel)
    {
        int j, k;

#pragma omp for schedule(static)
        // Sobel on the horizontal grid
        for(j = m; j < height - 1; j += m) {
            for(k = 1; k < width - 1; k++) {
                //printf("Thread %d with index j = %d and k = %d\n", omp_get_thread_num(), j, k);
                sobel_on_pixel(p, sobel, j, k, width, width);
            }
        }

#pragma omp for schedule(static)
        // Sobel on the vertical grid
        for(k = n; k < width - 1; k += n) {
            for(j = 1; j < height - 1; j++) {
                sobel_on_pixel(p, sobel, j, k, width, width);
            }
        }

#pragma omp for schedule(static)
        // Sobel inside the mini-blocks
        for(j = 0; j < height - 1; j += m) {
            for(k = 0; k < width - 1; k += n) {
                int w, h;

                if (j + m < height)
                    h = m + 1;
                else
                    h = height - j;

                if (k + n < width)
                    w = n + 1;
                else
                    w = width - k;

                sobel_filter_mini(p + j * width + k, w, h, width);
            }
        }

#pragma omp for schedule(static)
        // Update grid
        for(j = m; j < height - 1; j += m) {
            for(k = 1; k < width - 1; k++) {
                p[CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
                p[CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
                p[CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
            }
        }

#pragma omp for schedule(static)
        for(j = 1; j < height - 1; j++) {
            for(k = n; k < width - 1; k += n) {
                p[CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
                p[CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
                p[CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
            }
        }
    }

    free (sobel);
}

// Composition of filters -------------------------------------
void blur_filter_seq_with_defaults( pixel * p, int width, int height){
    blur_filter_seq(p, width, height, BLUR_SIZE, BLUR_THRESHOLD);
}

void complete_filter_seq( pixel * p, int width, int height) {
    gray_filter_seq(p, width, height);
    blur_filter_seq_with_defaults(p, width, height);
    sobel_filter_seq(p, width, height);
}

void complete_filter_omp( pixel * p, int width, int height) {
    gray_filter_omp(p, width, height);
    blur_filter_omp(p, width, height, BLUR_SIZE, BLUR_THRESHOLD);
    sobel_filter_omp(p, width, height);
}

// To be completed
void complete_filter_cuda( pixel * p, int width, int height) {
    gray_filter_cuda(p, width, height);
    blur_filter_seq_with_defaults(p, width, height);
    sobel_filter_omp(p, width, height);
}

// apply filter to sequence of Images ----------------------------
void bulk_apply_seq( pixel ** images, int * widths, int * heights, int n_images, void (*filter)(pixel*, int, int)){
    int i;
    for ( i = 0 ; i < n_images ; i++ )
    {
        (*filter)(images[i], widths[i], heights[i]);
    }
}

// Needs testing
void bulk_apply_omp( pixel ** images, int * widths, int * heights, int n_images, void (*filter)(pixel*, int, int)){
    int i;
#pragma omp for schedule(static)
    for ( i = 0 ; i < n_images ; i++ )
    {
        (*filter)(images[i], widths[i], heights[i]);
    }
}

void bulk_apply_cuda( pixel ** images, int * widths, int * heights, int n_images, void (*filter)(pixel*, int, int)){
     pixel* lin_images;
     int i;
     int n_pixels;
     n_pixels = 0;
     for( i = 0; i < n_images; i++){
        n_pixels += widths[i] * heights[i];
     }
     lin_images = (pixel*)malloc(n_pixels * sizeof(pixel));

     n_pixels = 0;
     for( i = 0; i < n_images; i++){
        memcpy(lin_images + n_pixels, images[i], widths[i] * heights[i] * sizeof(pixel));
        n_pixels += widths[i] * heights[i];
     }

     (*filter)(lin_images, n_pixels, 1);// cheating

     n_pixels = 0;
     for( i = 0; i < n_images; i++){
        memcpy(images[i], lin_images + n_pixels, widths[i] * heights[i] * sizeof(pixel));
        n_pixels += widths[i] * heights[i];
     }
}

// Applying filters to all images of Gif
void apply_to_all( animated_gif * image, void (*bulk_apply)(pixel**, int*, int*, int, void (*f)(pixel*, int, int)), void (*filter)(pixel*, int, int) )
{
    if(image)
        (*bulk_apply)(image->p, image->width, image->height, image->n_images, (*filter));
}

//------------------------ BEGIN OF MPI -------------------------------

//------------------------ BEGIN OF GROUPING -------------------------------

void set_MPI_comm_in_node(MPI_Comm * comm_in_node){
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, comm_in_node);
}


void set_MPI_comm_btwn_node(MPI_Comm comm_in_node, MPI_Comm * comm_btwn_nodes){
    int rank_in_node;
    MPI_Comm_rank(comm_in_node, &rank_in_node);
    int is_rank0, node_count;
    is_rank0 = (rank_in_node == 0) ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, is_rank0, 0, comm_btwn_nodes);
}

//------------------------ END OF GROUPING -------------------------------

void apply_to_all_MPI_stat( animated_gif * image, void (*filter)(pixel *, int, int) ){
    /*
       Shares the work among different nodes, via statical load balancing
       Master sends equal packages of work to all slave nodes and then performs the rest of the work himself
       Uses Isend and Irecv with waitall barrier afterwards
       */
    int l_id, g_id, s_id; // local_gif_id, global_gif_id, slave_id
    int rank_in_world, size_in_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_in_world);
    MPI_Comm_size(MPI_COMM_WORLD, &size_in_world);

    if(size_in_world == 1) return apply_to_all(image, bulk_apply_seq, (*filter));
    // From here on size_in_world > 1, and we use pure master slave architecture
    int n_slaves = size_in_world - 1;

    int height;
    int width;
    int n_images_global;
    if(rank_in_world == root_in_world) {
        n_images_global = image->n_images;
        width  = image->width[0];
        height = image->height[0];
    }

    MPI_Bcast(&n_images_global, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);

    // load_balancing: we share work in a bulk manner, not round-robin
    int* n_images_rank = malloc(size_in_world * sizeof(int));
    n_images_rank[root_in_world] = 0;
    for(s_id = 1; s_id < size_in_world; s_id++){
        n_images_rank[s_id] = n_images_global / n_slaves + (s_id <= (n_images_global % n_slaves));
    }// uses the fact that root_in_world = 0

    pixel * gif_local;

    int j;
    if(rank_in_world == root_in_world){
        g_id = 0;
        for(s_id = 1; s_id < size_in_world; s_id++){
            for(l_id = 0; l_id < n_images_rank[s_id]; l_id++, g_id++){
                MPI_Send(image->p[g_id], height * width, MPI_PIXEL, s_id, l_id, MPI_COMM_WORLD);
                MPI_Recv(image->p[g_id], height * width, MPI_PIXEL, s_id, l_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else{
        int n_images_local = n_images_rank[rank_in_world];
        gif_local = malloc(height * width * sizeof(pixel));
        for(l_id = 0; l_id < n_images_local; l_id++){
            MPI_Recv(gif_local, height * width, MPI_PIXEL, root_in_world, l_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            (*filter)(gif_local, width, height);
            MPI_Send(gif_local, height * width, MPI_PIXEL, root_in_world, l_id, MPI_COMM_WORLD);
        }
        free(gif_local);
    }
}

//------------------------ END OF MPI -------------------------------
//------------------------ BEGIN OF DEBUG TOOLS -------------------------------

pixel** reference_treated(pixel** p, int n_images, int width, int height){
    pixel** p_ref;
    p_ref = (pixel**)malloc(n_images * sizeof(pixel*));

    int i, j;
    for(i = 0; i < n_images; i++){
        p_ref[i] = (pixel*)malloc(width*height*sizeof(pixel));
        for(j = 0; j < width*height; j++){
            p_ref[i][j] = p[i][j];
        }
        complete_filter_seq(p_ref[i], width, height);
    }
    return p_ref;
}

void print_diff_with_ref(pixel** p, int n_images, int width, int height, pixel** p_ref){
    int i;
    for ( i = 0 ; i < n_images; i++ ) {
        int x, y, j;
        for(y = 0; y < height; y++){
            for(x = 0; x < width; x++){
                j = CONV(y, x, width);
                if(!eq_pixel(p[i][j], p_ref[i][j])){
                    printf("diff on img %3d in pixel (%3d,%3d): p_std = %d and p_new = %d\n", i, x, y, black_pixel(p_ref[i][j]), black_pixel(p[i][j]));
                }
            }
        }
    }
}

void hello_omp_mpi(MPI_Comm comm_in_node, MPI_Comm comm_btwn_nodes){
    int rank_in_world, size_in_world ;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank_in_world );
    MPI_Comm_size( MPI_COMM_WORLD, &size_in_world );
    int rank_in_node, size_in_node ;
    MPI_Comm_rank( comm_in_node, &rank_in_node );
    MPI_Comm_size( comm_in_node, &size_in_node );
    int rank_btwn_nodes, size_btwn_nodes ;
    //only in this case the master communicator is defined
    if(rank_in_node == 0){
        MPI_Comm_rank( comm_btwn_nodes, &rank_btwn_nodes );
        MPI_Comm_size( comm_btwn_nodes, &size_btwn_nodes );
    }
    else{
        rank_btwn_nodes = -1;
        size_btwn_nodes = -1;
    }
#pragma omp parallel
    {
        printf("Hello MPI:"
                "  world:"
                " %d (%d)"
                "  node"
                " %d (%d)"
                "  masters"
                " %d (%d)"
                "\t\tOpenMP:"
                " %d (%d)\n" ,
                rank_in_world, size_in_world,
                rank_in_node, size_in_node,
                rank_btwn_nodes, size_btwn_nodes,
                omp_get_thread_num(),
                omp_get_num_threads() ) ;
    }
}

bool is_constant_size_gif(animated_gif * image){
    int n_images, width, height;
    n_images = image->n_images;
    width = image->width[0];
    height = image->height[0];

    int i;
    for(i = 0; i < n_images; i++){
        int j;
        if(height != image->height[i] || width != image->width[i]){
            printf("WOW: your gif has varying dimensions\nfst image: (%d, %d)\nsnd image: (%d, %d)\n", width, height, image->width[i], image->height[i]);
            return false;
        }
    }
    return true;
}

//------------------------ END OF DEBUG TOOLS -------------------------------
double time_passed(struct timeval t1, struct timeval t2){
    return (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
}

int main( int argc, char ** argv )
{
    char * input_filename ;
    char * output_filename ;
    animated_gif * image ;
    struct timeval t1, t2;
    double duration ;

    int rc, rank_in_world, size_in_world;
    // Initializes MPI
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Type_contiguous(3, MPI_INT, &MPI_PIXEL);
    MPI_Type_commit(&MPI_PIXEL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_in_world);
    MPI_Comm_size(MPI_COMM_WORLD, &size_in_world);
    int n_images, height, width;

    MPI_Comm comm_in_node;
    set_MPI_comm_in_node(&comm_in_node);
    MPI_Comm comm_btwn_nodes;
    set_MPI_comm_btwn_node(comm_in_node, &comm_btwn_nodes);
    if(rank_in_world == root_in_world){
        if ( argc < 3 )
        {
            fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
            return 1 ;
        }

        input_filename = argv[1] ;
        output_filename = argv[2] ;

        // IMPORT Timer start
        gettimeofday(&t1, NULL);

        // Load file and store the pixels in array
        image = load_pixels( input_filename ) ;
        if ( image == NULL ) { return 1 ; }

        // IMPORT Timer stop
        gettimeofday(&t2, NULL);

        printf( "GIF loaded from file %s with %d image(s) in %lf s\n",
                input_filename, image->n_images, time_passed(t1, t2) ) ;
        if(!is_constant_size_gif(image)){
            MPI_Finalize();
            return 1;
        }

        printf( "GIF STATS: width = %d, height = %d, number of images = %d\n", image->height[0], image->width[0], image->n_images);

        // FILTER Timer start
        gettimeofday(&t1, NULL);
    }

        bulk_apply_cuda(image->p, image->width, image->height, image->n_images, gray_filter_cuda);


    if(rank_in_world == root_in_world){
        gettimeofday(&t2, NULL);

        printf( "SOBEL done in %lf s\n", time_passed(t1, t2) ) ;

        gettimeofday(&t1, NULL);

        // Store file from array of pixels to GIF file
        if ( !store_pixels( output_filename, image ) ) { return 1 ; }

        gettimeofday(&t2, NULL);

        printf( "Export done in %lf s in file %s\n--------------------------------------\n", time_passed(t1, t2), output_filename ) ;

    }

    MPI_Finalize();
    return 0 ;
}
