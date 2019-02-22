// void apply_gray_line( animated_gif * image )
// {
//     int i, j, k ;
//     pixel ** p ;
//
//     p = image->p ;
//
//     for ( i = 0 ; i < image->n_images ; i++ )
//     {
//         for ( j = 0 ; j < 10 ; j++ )
//         {
//             for ( k = image->width[i]/2 ; k < image->width[i] ; k++ )
//             {
//                 p[i][CONV(j,k,image->width[i])].r = 0 ;
//                 p[i][CONV(j,k,image->width[i])].g = 0 ;
//                 p[i][CONV(j,k,image->width[i])].b = 0 ;
//             }
//         }
//     }
// }

// void apply_to_all_MPI_stat_scatter( animated_gif * image, void (*filter)(pixel *, int, int) ){
//     int rank_in_world, size_in_world;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank_in_world);
//     MPI_Comm_size(MPI_COMM_WORLD, &size_in_world);
//
//     int n_images_global;
//     if(rank_in_world == root_in_world) n_images_global =  image->n_images;
//     MPI_Bcast(&n_images_global, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);
//
//     int height, width;
//     if(rank_in_world == root_in_world) height = *(image->height), width = *(image->width);
//     MPI_Bcast(&height, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);
//     MPI_Bcast(&width, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);
//
//     // ----------------------- Scatter Static Load Balancing ---------------------------
//     // Some problems:
//     //  - Very non-uniform images, may have different heights and widths, would have to create a large buffer and deal with the whole logic to cut it
//     //  - Needs loads of RAM, as it creates a compy of image for rank 0
//     int n_images_local = n_images_global / size_in_world + (n_images_global % size_in_world > rank_in_world);
//
//     pixel * gif_buffer;
//     if(rank_in_world == root_in_world) gif_buffer = malloc(n_images_global * height * width * sizeof(pixel)); // be careful with this, what if root is no 0
//     pixel * local_gif_buffer;
//     if(rank_in_world < N_GIF_MASTERS) local_gif_buffer = malloc(n_images_local * height * width * sizeof(pixel));
//
//
//     MPI_Scatter(gif_buffer, n_images_global * height* width, MPI_PIXEL, local_gif_buffer, n_images_local * height * width, MPI_PIXEL, root_in_world, MPI_COMM_WORLD);
//
//     for(int i = 0; i < n_images_local; i++){
//         gray_filter(local_gif_buffer + i, height, width);
//     }
//
//     MPI_Gather(local_gif_buffer, n_images_local * height * width, MPI_PIXEL, gif_buffer, n_images_global * height * width, MPI_PIXEL, root_in_world, MPI_COMM_WORLD);
//
// }

// void apply_to_all_MPI_dyn( animated_gif * image, void (*filter)(pixel *, int, int) ){
//     int rank_in_world, size_in_world;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank_in_world);
//     MPI_Comm_size(MPI_COMM_WORLD, &size_in_world);
//
//     int n_images_global;
//     if(rank_in_world == root_in_world) n_images_global =  image->n_images;
//     MPI_Bcast(&n_images_global, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);
//
//     int height, width;
//     if(rank_in_world == root_in_world) height = *(image->height), width = *(image->width);
//     MPI_Bcast(&height, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);
//     MPI_Bcast(&width, 1, MPI_INT, root_in_world, MPI_COMM_WORLD);
//
//     // ----------------------- Dynamic Load Balancing ---------------------------
//     // Load Balancing variables
//     // if there are more slaves than jobs, set the rest free
//     int num_slaves = min(N_GIF_MASTERS, n_images_global);
//     if(num_slaves == 0) return apply_to_all(image, filter); // master/slave with no slave then just run it sequentialy with master
//
//     MPI_Request * l_req;
//     pixel * image_buffer;
//     if(rank_in_world == root_in_world){
//         l_req = (MPI_Request*)malloc(num_slaves*sizeof(MPI_Request));
//         // spawn work to all slaves
//         for(int j = 0; j < num_slaves; j++){
//             image_buffer = image->p[j];
//             MPI_Ssend(image_buffer, height * width, MPI_PIXEL, j, 0, MPI_COMM_WORLD); // This is fucking weird Send gives different results from Ssend
//             MPI_Irecv(image->p[j], height * width, MPI_PIXEL, j, 1, MPI_COMM_WORLD, l_req + j);
//         }
//         // spawn remaining work to available slaves
//         for(int free_slave, is_free = 0, j = num_slaves; j < n_images_global; j++){
//             image_buffer = image->p[j];
//             do {// Loop util someone frees up
//                 MPI_Testany(num_slaves, l_req, &free_slave, &is_free, MPI_STATUS_IGNORE);
//             } while(is_free == 0);
//             MPI_Ssend(image_buffer, height * width, MPI_PIXEL, free_slave, 0, MPI_COMM_WORLD); // This is fucking weird Send gives different results from Ssend
//             MPI_Irecv(image->p[j], height * width, MPI_PIXEL, free_slave, 1, MPI_COMM_WORLD, l_req + free_slave);
//         }
//         // Waits the completion of all task then tell them the work is finished
//         MPI_Waitall(num_slaves, l_req, MPI_STATUSES_IGNORE);
//
//         free(l_req);
//     } else if(rank_in_world < N_GIF_MASTERS){
//         MPI_Status stat;
//
//         MPI_Recv(image_buffer, height * width, MPI_PIXEL, root_in_world, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
//         while(stat.MPI_TAG == 0){
//
//             MPI_Send(&m, 1, MPI_PIXEL, root_in_world, 1, MPI_COMM_WORLD);
//             MPI_Recv(image_buffer, N, MPI_PIXEL, root_in_world, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
//         }
//
//         free(image_buffer);
//     }
// }

