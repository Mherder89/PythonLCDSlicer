def makeImage(sparse_array_to_plot, sparse_img_coords, fn, subtractMin = False):
    d_out_img = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.uint8) # final slicer image
    d_sparse_scaled = cuda.device_array(int(sparse_array_to_plot.shape[0]), dtype=np.float32) 

    threadsperblock = (32, 32) 
    blockspergrid_x = math.ceil(d_out_img.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(d_out_img.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    initMatrix[blockspergrid, threadsperblock](d_out_img,0) #init matrix on GPU

    result = np.zeros(1, dtype=np.float64)
    threadsperblock = 128 # max for rtx2070
    blockspergrid = (sparse_array_to_plot.shape[0] + (threadsperblock - 1)) // threadsperblock
    get_max[blockspergrid,threadsperblock](result, sparse_array_to_plot)
    array_max = result[0]
    get_min[blockspergrid,threadsperblock](result, sparse_array_to_plot)
    array_min = result[0]
    print(array_min, array_max)
    
    threadsperblock = 128 #1024 # max for rtx2070
    blockspergrid = (d_sparse_scaled.shape[0] + (threadsperblock - 1)) // threadsperblock
    if (subtractMin):
        normTo255[blockspergrid, threadsperblock](d_sparse_scaled, sparse_array_to_plot, array_min, array_max)
    else:
        normTo255[blockspergrid, threadsperblock](d_sparse_scaled, sparse_array_to_plot, 0, array_max)

    threadsperblock = 128 #1024 # max for rtx2070
    blockspergrid = (sparse_array_to_plot.shape[0] + (threadsperblock - 1)) // threadsperblock
    sparse_to_img[blockspergrid, threadsperblock](d_out_img, d_sparse_scaled, sparse_img_coords)
    out_img = d_out_img.copy_to_host(stream=stream)
    stream.synchronize()
    plt.imsave(fname=fn, arr=out_img, cmap='gray_r', format='png')

