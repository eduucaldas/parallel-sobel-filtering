#ifndef Cuda_filters_h
#define Cuda_filters_h

#include "gif_io.h"

void gray_filter_cuda(pixel* p, int width, int height);

void blur_filter_cuda(pixel* p, int width, int height, int size, int threshold);

#endif
