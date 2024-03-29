# Efficient Sobel Filtering of gif with MPI, openMP and CUDA
=============================================================

## Quick Description
This application uses a stencil-based scheme to apply a filter to an existing image or set of images. It means that the main process of this code is a traversal of each pixel of the image and the application of a 2D stencil. The parallelism will therefore come from the ability to split the image into multiple pieces to process it faster. Another step beyond might be to exploit the parallelism between the images.

## Project structure
- images
    - original: the original gifs before any treated
    - original_processed: the gifs after sequential treatment, used on regression testing
    - processed: processed gifs, using the amount of parallelisation you want
    - original-bad: gifs in a bad format, i.e. with pictures of varying size

## How to use
Now that we implemented the MPI and openMP solutions we can run the script run_test_OMPI.sh to apply the sobel filter on all the gifs in the images/original folder
This will produce the results in the images/processed folder

## Examples
TODO

## Benchmarking
TODO

## TODO
- [x] Separate files, main is too big
- [x] mix mpi with openmp
- [x] implement: OMP by splitting image
    + [x] gray\_filter
    + [x] blur\_filter
    + [x] sobel\_filter
- [x] implement: CUDA by splitting image
    + [x] gray\_filter
    + [x] blur\_filter
    + [x] sobel\_filter
- [x] adapt: bulk\_MPI to treat gifs with different height and width
- [x] optimise: bulk\_MPI, send many images and then treat them with some other bulk
- [ ] implement: bulk\_MPI to use slaves without gpu
- [ ] implement: MPI by splitting image
- [ ] implement: bulk\_OMP
- [ ] optimise: OMP, use dynamic instead of static
- [ ] implement: bulk images to treat on CUDA
    + [x] gray\_filter
    + [ ] blur\_filter
    + [ ] sobel\_filter
- [ ] benchmarking framework
    + [ ] stats of the run: MPI\_nodes, MPI\_tasks, OMP\threads, input stats
    + [ ] fprint the stats
    + [ ] command-line arguments regarding hardware
- [ ] load-balancing

