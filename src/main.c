/*
 * INF560
 *
 * Image Filtering Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <mpi.h>

#include <gif_lib.h>

#include <omp.h>

#define SOBELF_DEBUG 0

/* Represent one pixel from the image */
typedef struct pixel
{
    int r ; /* Red */
    int g ; /* Green */
    int b ; /* Blue */
} pixel ;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
    int n_images ; /* Number of images */
    int * width ; /* Width of each image */
    int * height ; /* Height of each image */
    pixel ** p ; /* Pixels of each image */
    GifFileType * g ; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif ;

MPI_Datatype MPI_PIXEL;
/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif * load_pixels( char * filename )
{
    GifFileType * g ;
    ColorMapObject * colmap ;
    int error ;
    int n_images ;
    int * width ;
    int * height ;
    pixel ** p ;
    int i ;
    animated_gif * image ;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName( filename, &error ) ;
    if ( g == NULL )
    {
        fprintf( stderr, "Error DGifOpenFileName %s\n", filename ) ;
        return NULL ;
    }

    /* Read the GIF image */
    error = DGifSlurp( g ) ;
    if ( error != GIF_OK )
    {
        fprintf( stderr,
                "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error) ) ;
        return NULL ;
    }

    /* Grab the number of images and the size of each image */
    n_images = g->ImageCount ;

    width = (int *)malloc( n_images * sizeof( int ) ) ;
    if ( width == NULL )
    {
        fprintf( stderr, "Unable to allocate width of size %d\n",
                n_images ) ;
        return 0 ;
    }

    height = (int *)malloc( n_images * sizeof( int ) ) ;
    if ( height == NULL )
    {
        fprintf( stderr, "Unable to allocate height of size %d\n",
                n_images ) ;
        return 0 ;
    }

    /* Fill the width and height */
    for ( i = 0 ; i < n_images ; i++ )
    {
        width[i] = g->SavedImages[i].ImageDesc.Width ;
        height[i] = g->SavedImages[i].ImageDesc.Height ;

#if SOBELF_DEBUG
        printf( "Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
                i,
                g->SavedImages[i].ImageDesc.Left,
                g->SavedImages[i].ImageDesc.Top,
                g->SavedImages[i].ImageDesc.Width,
                g->SavedImages[i].ImageDesc.Height,
                g->SavedImages[i].ImageDesc.Interlace,
                g->SavedImages[i].ImageDesc.ColorMap
              ) ;
#endif
    }


    /* Get the global colormap */
    colmap = g->SColorMap ;
    if ( colmap == NULL )
    {
        fprintf( stderr, "Error global colormap is NULL\n" ) ;
        return NULL ;
    }

#if SOBELF_DEBUG
    printf( "Global CM: count:%d bpp:%d sort:%d\n",
            g->SColorMap->ColorCount,
            g->SColorMap->BitsPerPixel,
            g->SColorMap->SortFlag
          ) ;
#endif

    /* Allocate the array of pixels to be returned */
    p = (pixel **)malloc( n_images * sizeof( pixel * ) ) ;
    if ( p == NULL )
    {
        fprintf( stderr, "Unable to allocate array of %d images\n",
                n_images ) ;
        return NULL ;
    }

    for ( i = 0 ; i < n_images ; i++ )
    {
        p[i] = (pixel *)malloc( width[i] * height[i] * sizeof( pixel ) ) ;
        if ( p[i] == NULL )
        {
            fprintf( stderr, "Unable to allocate %d-th array of %d pixels\n",
                    i, width[i] * height[i] ) ;
            return NULL ;
        }
    }

    /* Fill pixels */

    /* For each image */
    for ( i = 0 ; i < n_images ; i++ )
    {
        int j ;

        /* Get the local colormap if needed */
        if ( g->SavedImages[i].ImageDesc.ColorMap )
        {

            /* TODO No support for local color map */
            fprintf( stderr, "Error: application does not support local colormap\n" ) ;
            return NULL ;

            colmap = g->SavedImages[i].ImageDesc.ColorMap ;
        }

        /* Traverse the image and fill pixels */
        for ( j = 0 ; j < width[i] * height[i] ; j++ )
        {
            int c ;

            c = g->SavedImages[i].RasterBits[j] ;

            p[i][j].r = colmap->Colors[c].Red ;
            p[i][j].g = colmap->Colors[c].Green ;
            p[i][j].b = colmap->Colors[c].Blue ;
        }
    }

    /* Allocate image info */
    image = (animated_gif *)malloc( sizeof(animated_gif) ) ;
    if ( image == NULL )
    {
        fprintf( stderr, "Unable to allocate memory for animated_gif\n" ) ;
        return NULL ;
    }

    /* Fill image fields */
    image->n_images = n_images ;
    image->width = width ;
    image->height = height ;
    image->p = p ;
    image->g = g ;

#if SOBELF_DEBUG
    printf( "-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0] ) ;
#endif

    return image ;
}

int output_modified_read_gif( char * filename, GifFileType * g )
{
    GifFileType * g2 ;
    int error2 ;

#if SOBELF_DEBUG
    printf( "Starting output to file %s\n", filename ) ;
#endif

    g2 = EGifOpenFileName( filename, false, &error2 ) ;
    if ( g2 == NULL )
    {
        fprintf( stderr, "Error EGifOpenFileName %s\n",
                filename ) ;
        return 0 ;
    }

    g2->SWidth = g->SWidth ;
    g2->SHeight = g->SHeight ;
    g2->SColorResolution = g->SColorResolution ;
    g2->SBackGroundColor = g->SBackGroundColor ;
    g2->AspectByte = g->AspectByte ;
    g2->SColorMap = g->SColorMap ;
    g2->ImageCount = g->ImageCount ;
    g2->SavedImages = g->SavedImages ;
    g2->ExtensionBlockCount = g->ExtensionBlockCount ;
    g2->ExtensionBlocks = g->ExtensionBlocks ;

    error2 = EGifSpew( g2 ) ;
    if ( error2 != GIF_OK )
    {
        fprintf( stderr, "Error after writing g2: %d <%s>\n",
                error2, GifErrorString(g2->Error) ) ;
        return 0 ;
    }

    return 1 ;
}


int store_pixels( char * filename, animated_gif * image )
{
    int n_colors = 0 ;
    pixel ** p ;
    int i, j, k ;
    GifColorType * colormap ;

    /* Initialize the new set of colors */
    colormap = (GifColorType *)malloc( 256 * sizeof( GifColorType ) ) ;
    if ( colormap == NULL )
    {
        fprintf( stderr,
                "Unable to allocate 256 colors\n" ) ;
        return 0 ;
    }

    /* Everything is white by default */
    for ( i = 0 ; i < 256 ; i++ )
    {
        colormap[i].Red = 255 ;
        colormap[i].Green = 255 ;
        colormap[i].Blue = 255 ;
    }

    /* Change the background color and store it */
    int moy ;
    moy = (
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue
          )/3 ;
    if ( moy < 0 ) moy = 0 ;
    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
    printf( "[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue,
            moy, moy, moy ) ;
#endif

    colormap[0].Red = moy ;
    colormap[0].Green = moy ;
    colormap[0].Blue = moy ;

    image->g->SBackGroundColor = 0 ;

    n_colors++ ;

    /* Process extension blocks in main structure */
    for ( j = 0 ; j < image->g->ExtensionBlockCount ; j++ )
    {
        int f ;

        f = image->g->ExtensionBlocks[j].Function ;
        if ( f == GRAPHICS_EXT_FUNC_CODE )
        {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3] ;

            if ( tr_color >= 0 &&
                    tr_color < 255 )
            {

                int found = -1 ;

                moy =
                    (
                     image->g->SColorMap->Colors[ tr_color ].Red
                     +
                     image->g->SColorMap->Colors[ tr_color ].Green
                     +
                     image->g->SColorMap->Colors[ tr_color ].Blue
                    ) / 3 ;
                if ( moy < 0 ) moy = 0 ;
                if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[ tr_color ].Red,
                        image->g->SColorMap->Colors[ tr_color ].Green,
                        image->g->SColorMap->Colors[ tr_color ].Blue,
                        moy, moy, moy ) ;
#endif

                for ( k = 0 ; k < n_colors ; k++ )
                {
                    if (
                            moy == colormap[k].Red
                            &&
                            moy == colormap[k].Green
                            &&
                            moy == colormap[k].Blue
                       )
                    {
                        found = k ;
                    }
                }
                if ( found == -1  )
                {
                    if ( n_colors >= 256 )
                    {
                        fprintf( stderr,
                                "Error: Found too many colors inside the image\n"
                               ) ;
                        return 0 ;
                    }

#if SOBELF_DEBUG
                    printf( "[DEBUG]\tNew color %d\n",
                            n_colors ) ;
#endif

                    colormap[n_colors].Red = moy ;
                    colormap[n_colors].Green = moy ;
                    colormap[n_colors].Blue = moy ;


                    image->g->ExtensionBlocks[j].Bytes[3] = n_colors ;

                    n_colors++ ;
                } else
                {
#if SOBELF_DEBUG
                    printf( "[DEBUG]\tFound existing color %d\n",
                            found ) ;
#endif
                    image->g->ExtensionBlocks[j].Bytes[3] = found ;
                }
            }
        }
    }

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->g->SavedImages[i].ExtensionBlockCount ; j++ )
        {
            int f ;

            f = image->g->SavedImages[i].ExtensionBlocks[j].Function ;
            if ( f == GRAPHICS_EXT_FUNC_CODE )
            {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] ;

                if ( tr_color >= 0 &&
                        tr_color < 255 )
                {

                    int found = -1 ;

                    moy =
                        (
                         image->g->SColorMap->Colors[ tr_color ].Red
                         +
                         image->g->SColorMap->Colors[ tr_color ].Green
                         +
                         image->g->SColorMap->Colors[ tr_color ].Blue
                        ) / 3 ;
                    if ( moy < 0 ) moy = 0 ;
                    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                    printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                            i,
                            image->g->SColorMap->Colors[ tr_color ].Red,
                            image->g->SColorMap->Colors[ tr_color ].Green,
                            image->g->SColorMap->Colors[ tr_color ].Blue,
                            moy, moy, moy ) ;
#endif

                    for ( k = 0 ; k < n_colors ; k++ )
                    {
                        if (
                                moy == colormap[k].Red
                                &&
                                moy == colormap[k].Green
                                &&
                                moy == colormap[k].Blue
                           )
                        {
                            found = k ;
                        }
                    }
                    if ( found == -1  )
                    {
                        if ( n_colors >= 256 )
                        {
                            fprintf( stderr,
                                    "Error: Found too many colors inside the image\n"
                                   ) ;
                            return 0 ;
                        }

#if SOBELF_DEBUG
                        printf( "[DEBUG]\tNew color %d\n",
                                n_colors ) ;
#endif

                        colormap[n_colors].Red = moy ;
                        colormap[n_colors].Green = moy ;
                        colormap[n_colors].Blue = moy ;


                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors ;

                        n_colors++ ;
                    } else
                    {
#if SOBELF_DEBUG
                        printf( "[DEBUG]\tFound existing color %d\n",
                                found ) ;
#endif
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found ;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf( "[DEBUG] Number of colors after background and transparency: %d\n",
            n_colors ) ;
#endif

    p = image->p ;

    /* Find the number of colors inside the image */
    for ( i = 0 ; i < image->n_images ; i++ )
    {

#if SOBELF_DEBUG
        printf( "OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
                i, image->n_images, image->width[i], image->height[i] ) ;
#endif

        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ )
        {
            int found = 0 ;
            for ( k = 0 ; k < n_colors ; k++ )
            {
                if ( p[i][j].r == colormap[k].Red &&
                        p[i][j].g == colormap[k].Green &&
                        p[i][j].b == colormap[k].Blue )
                {
                    found = 1 ;
                }
            }

            if ( found == 0 )
            {
                if ( n_colors >= 256 )
                {
                    fprintf( stderr,
                            "Error: Found too many colors inside the image\n"
                           ) ;
                    return 0 ;
                }

#if SOBELF_DEBUG
                printf( "[DEBUG] Found new %d color (%d,%d,%d)\n",
                        n_colors, p[i][j].r, p[i][j].g, p[i][j].b ) ;
#endif

                colormap[n_colors].Red = p[i][j].r ;
                colormap[n_colors].Green = p[i][j].g ;
                colormap[n_colors].Blue = p[i][j].b ;
                n_colors++ ;
            }
        }
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: found %d color(s)\n", n_colors ) ;
#endif


    /* Round up to a power of 2 */
    if ( n_colors != (1 << GifBitSize(n_colors) ) )
    {
        n_colors = (1 << GifBitSize(n_colors) ) ;
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: Rounding up to %d color(s)\n", n_colors ) ;
#endif

    /* Change the color map inside the animated gif */
    ColorMapObject * cmo ;

    cmo = GifMakeMapObject( n_colors, colormap ) ;
    if ( cmo == NULL )
    {
        fprintf( stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
                n_colors ) ;
        return 0 ;
    }

    image->g->SColorMap = cmo ;

    /* Update the raster bits according to color map */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ )
        {
            int found_index = -1 ;
            for ( k = 0 ; k < n_colors ; k++ )
            {
                if ( p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                        p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                        p[i][j].b == image->g->SColorMap->Colors[k].Blue )
                {
                    found_index = k ;
                }
            }

            if ( found_index == -1 )
            {
                fprintf( stderr,
                        "Error: Unable to find a pixel in the color map\n" ) ;
                return 0 ;
            }

            image->g->SavedImages[i].RasterBits[j] = found_index ;
        }
    }


    /* Write the final image */
    if ( !output_modified_read_gif( filename, image->g ) ) { return 0 ; }

    return 1 ;
}

//------------------------ END OF FILE TREATING -------------------------------

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

// Configuration Variables
#define BLUR_SIZE 5
#define BLUR_THRESHOLD 20
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

// needs the same change as in blur_filter_seq_seq
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

            /* Copy the middle part of the image */
#pragma omp for schedule(static)
            for(j=height/10-size; j<j_end; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    new[CONV(j,k,width)].r = p[CONV(j,k,width)].r ;
                    new[CONV(j,k,width)].g = p[CONV(j,k,width)].g ;
                    new[CONV(j,k,width)].b = p[CONV(j,k,width)].b ;
                }
            }

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
    blur_filter_seq_with_defaults(p, width, height);
    sobel_filter_omp(p, width, height);
}

// apply filter to sequence of Images ----------------------------
void bulk_apply_seq( pixel **images, int *widths, int *heights, int n_images, void (*filter)(pixel*, int, int)){
    int i;
    for ( i = 0 ; i < n_images ; i++ )
    {
        (*filter)(images[i], widths[i], heights[i]);
    }
}

// Applying filters to all images of Gif
void apply_to_all( animated_gif * image, void (*bulk_apply)(pixel**, int*, int*, int, void (*f)(pixel*, int, int)), void (*filter)(pixel*, int, int) )
{
    (*bulk_apply)(image->p, image->width, image->height, image->n_images, (*filter));
}

//------------------------ BEGIN OF MPI -------------------------------

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

void hello_omp_mpi(){
    int mpi_rank, mpi_size ;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_size ) ;
#pragma omp parallel
    {
        printf("Hello MPI %d (%d) & OpenMP %d (%d)\n",mpi_rank, mpi_size,
                omp_get_thread_num(),
                omp_get_num_threads() ) ;
    }
}

//------------------------ END OF DEBUG TOOLS -------------------------------

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
    int n_images,height, width;
    pixel** p_original;

    if(rank_in_world == root_in_world){
        if ( argc < 3 )
        {
            fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
            return 1 ;
        }

        input_filename = argv[1] ;
        output_filename = argv[2] ;

        /* IMPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Load file and store the pixels in array */
        image = load_pixels( input_filename ) ;
        if ( image == NULL ) { return 1 ; }

        /* IMPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "GIF loaded from file %s with %d image(s) in %lf s\n",
                input_filename, image->n_images, duration ) ;
        int i;
        n_images = image->n_images;
        width = image->width[0];
        height = image->height[0];
        for(i = 0; i < n_images; i++){
            int j;
            if(height != image->height[i] || width != image->width[i]){
                printf("WOW: your gif has varying dimensions\nfst image: (%d, %d)\nsnd image: (%d, %d)\n", width, height, image->width[i], image->height[i]);
                MPI_Finalize();
                return 1;
            }
        }
        printf( "GIF STATS: width = %d, height = %d, number of images = %d\n", image->height[0], image->width[0], image->n_images);

        /* FILTER Timer start */
        gettimeofday(&t1, NULL);
    }

    apply_to_all_MPI_stat(image, complete_filter_omp);

    if(rank_in_world == root_in_world){
        int i;

        /* FILTER Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "SOBEL done in %lf s\n", duration ) ;

        /* EXPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Store file from array of pixels to GIF file */
        if ( !store_pixels( output_filename, image ) ) { return 1 ; }

        /* EXPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "Export done in %lf s in file %s\n--------------------------------------\n", duration, output_filename ) ;

    }

    MPI_Finalize();
    return 0 ;
}
