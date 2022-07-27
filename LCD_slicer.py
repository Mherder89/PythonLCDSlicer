import trimesh
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
cuda.select_device(0)
stream = cuda.stream()


filepath = 'C:/Resin/articulated-dragon-mcgybeer20211115-748-6f8p9f/mcgybeer/articulated-dragon-mcgybeer/Dragon_v2/' #'C:/VSCode/PythonTest/' #
filename = 'Dragon_v2.stl' #'test.stl'# 'cube.stl'#
mesh_raw = trimesh.load_mesh(filepath + filename)
# give it a color
mesh_raw.visual.face_colors = [100, 100, 100, 255]
# and show it
#mesh_raw.show(viewer='gl')


origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
mesh = mesh_raw.copy()
transform, bounds = trimesh.bounds.oriented_bounds(mesh, angle_digits=1, ordered=False, normal=zaxis)
#mesh.apply_transform(trimesh.transformations.reflection_matrix(bounds/2.0, zaxis)) 
scale_rel_to_buildplate = 0.5
scale = min([3840.0,2400.0,4000.0]/bounds)*scale_rel_to_buildplate # min([10,10,10]/bounds) #
# voxel_bounds = scale*bounds
# voxel_bounds = np.ceil(voxel_bounds)
mesh.apply_transform(transform)
mesh.apply_transform(trimesh.transformations.translation_matrix(bounds/2.0))
mesh.apply_transform(trimesh.transformations.scale_matrix(scale, origin))
# print(voxel_bounds)
# print(mesh.bounding_box.bounds)
voxel_bounds = [int(np.ceil(mesh.bounding_box.bounds[1,0])), int(np.ceil(mesh.bounding_box.bounds[1,1])), int(np.ceil(mesh.bounding_box.bounds[1,2]))]
print(voxel_bounds)
voxel_bounds[0] += 5 # 2px left 3px right border
voxel_bounds[1] += 5 
voxel_bounds[0] += voxel_bounds[0] % 2
voxel_bounds[1] += voxel_bounds[1] % 2
mesh.apply_transform(trimesh.transformations.translation_matrix([2.5,2.5,0])) # shift xy by half a pixel (+2px Boder), so pixel index coordinate is mid pixel
z_layer = 10
project_z0_matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,-z_layer],[0,0,0,1]]
slice = mesh.section(plane_origin=[0,0,z_layer], plane_normal=zaxis)
slice_2D, to_3D = slice.to_planar(to_2D=project_z0_matrix, normal=zaxis, check=False)
# filledImg = slice_2D.rasterize(pitch=1.0, origin=[0,0], resolution=voxel_bounds[0:2], fill=True, width=0)
# from PIL import ImageChops
# # borderIMG = slice_2D.rasterize(pitch=1.0, origin=[0,0], resolution=voxel_bounds[0:2], fill=False, width=0)
# # binaryImg = ImageChops.difference(filledImg,borderIMG) 
# # sadly this does not work, Pixels that are too close are discarded later but take up resources :-(
# # need slicer for only inner Pixels, no touching!
# binaryImg = filledImg
# print(slice_2D.plot_entities())
# sparseImg = np.nonzero(binaryImg)
# sparseImg = np.transpose([sparseImg[1],sparseImg[0]])

# slice
d_sliceImg = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.int32)
threadsperblock = (32, 32) 
blockspergrid_x = math.ceil(d_sliceImg.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(d_sliceImg.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
initMatrix[blockspergrid, threadsperblock](d_sliceImg, 0)

for i in range(slice_2D.polygons_closed.shape[0]):
    h_path = np.asarray(list(slice_2D.polygons_closed[i].exterior.coords), dtype=np.float64)
    d_path = cuda.to_device(h_path)
    count_crossings[blockspergrid, threadsperblock](d_sliceImg, d_path)

matrix_mod2[blockspergrid, threadsperblock](d_sliceImg)
out_img = d_sliceImg.copy_to_host(stream=stream)
stream.synchronize()
plt.imsave(fname='custom_slice.png', arr=out_img, cmap='gray_r', format='png')


#convert to sparse form
d_sliceImg = d_sliceImg.reshape(d_sliceImg.shape[0]*d_sliceImg.shape[1])
scan_array(d_sliceImg)
d_sliceImg_scan_limit = cuda.device_array(1, d_sliceImg.dtype)
# d_sliceImg = d_sliceImg.reshape((int(voxel_bounds[0]),int(voxel_bounds[1]))) # better to discard cause of large int32
get_limit[1, 1](d_sliceImg, d_sliceImg_scan_limit)
slice_img_scan_limit = int(d_sliceImg_scan_limit.copy_to_host(stream=stream)[0])
stream.synchronize()
# print(slice_img_scan_limit)
d_sparseImg = cuda.device_array((slice_img_scan_limit,2), dtype=np.int32)#reduced_points_mask_size
threadsperblock = 128 #1024 # max for rtx2070
blockspergrid = (d_sliceImg.shape[0] + (threadsperblock - 1)) // threadsperblock
sliceImg_to_sparseImg[blockspergrid, threadsperblock](d_sliceImg, d_sparseImg, voxel_bounds[1])
# debug
# d_out_img = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.uint8) # final slicer image
# threadsperblock = (32, 32) 
# blockspergrid_x = math.ceil(d_out_img.shape[0] / threadsperblock[0])
# blockspergrid_y = math.ceil(d_out_img.shape[1] / threadsperblock[1])
# blockspergrid = (blockspergrid_x, blockspergrid_y)
# initMatrix[blockspergrid, threadsperblock](d_out_img,0) #init matrix on GPU
# threadsperblock = 128 #1024 # max for rtx2070
# blockspergrid = (d_sparseImg.shape[0] + (threadsperblock - 1)) // threadsperblock
# sparsePixel_to_img[blockspergrid, threadsperblock](d_out_img, d_sparseImg)
# out_img = d_out_img.copy_to_host(stream=stream)
# stream.synchronize()
# plt.imsave(fname='custom_slice_sparse.png', arr=out_img, cmap='gray_r', format='png')


# get closest point on faces
# copy data to device
sliceme = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[0, 0, -1], plane_origin=[0,0,z_layer])
sliceme = trimesh.intersections.slice_mesh_plane(sliceme, plane_normal=[0, 0, 1], plane_origin=[0,0,z_layer-3])
#sliceme.bounding_box.bounds
# print(sliceme.faces.shape)
sliceme.remove_degenerate_faces(height=1e-03) # remove and face with one side < height. < 1e-4 still has errors
# print(sliceme.faces.shape)
#sliceme.show(viewer='gl', flags={'wireframe': True})
faces = np.asarray(sliceme.faces, dtype=np.uint32)#np.asarray(error_faces, dtype=np.uint32) #
vertices = np.asarray(sliceme.vertices, dtype=np.float32)
sortObtuseAngleToC(faces,vertices)

# d_sparseImg = cuda.to_device(np.asarray(sparseImg, dtype=np.int32), stream=stream) # list of pixels inside (or at border) of the mesh
d_faces = cuda.to_device(faces, stream=stream) # faces of the cut mesh. Each is a index reference to 3 vertices.
d_vertices = cuda.to_device(vertices, stream=stream) # vertices of the cut mesh. Lowest z coord has to be zIndex.
d_points_on_surface = cuda.device_array((d_sparseImg.shape[0],4), np.float32) # output: array of closes points on any face for input point in sparseImg
stream.synchronize()

#calc closest points
threadsperblock = 32 #1024 # max for rtx2070
blockspergrid = (d_points_on_surface.shape[0] + (threadsperblock - 1)) // threadsperblock
findClosesPoints[blockspergrid, threadsperblock](d_points_on_surface, d_sparseImg, z_layer, d_faces, d_vertices)
# points_on_surface = d_points_on_surface.copy_to_host(stream=stream)
# stream.synchronize()
# points_on_surface.shape[0]


d_out_img = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.int32)
threadsperblock = (32, 32) 
blockspergrid_x = math.ceil(d_out_img.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(d_out_img.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
initMatrix[blockspergrid, threadsperblock](d_out_img,0)
threadsperblock = 128 #1024 # max for rtx2070
blockspergrid = (d_points_on_surface.shape[0] + (threadsperblock - 1)) // threadsperblock
mark_filling_points[blockspergrid, threadsperblock](d_out_img, d_points_on_surface, d_sparseImg)
# out_img = d_out_img.copy_to_host(stream=stream)
# stream.synchronize()
# plt.imsave(fname='filling_pixels.png', arr=out_img, cmap='gray_r', format='png')


# reduce d_points_on_surface: extract relevant points ie < max_distance
# generate a binary mask for relevant points
d_points_mask = cuda.device_array(int(d_points_on_surface.shape[0]+d_points_on_surface.shape[0]%2), dtype=np.uint32) # scan algo needs even size
threadsperblock = 32 #1024 # max for rtx2070
blockspergrid = (d_points_mask.shape[0] + (threadsperblock - 1)) // threadsperblock
get_active_points[blockspergrid, threadsperblock](d_points_mask, d_points_on_surface)

#scan mask array and extract largest value as reduced array size
scan_array(d_points_mask)
d_mask_limit = cuda.device_array(2, np.uint32)
get_mask_limit[1, 1](d_points_mask, d_mask_limit)
mask_limit = d_mask_limit.copy_to_host(stream=stream)
stream.synchronize()
reduced_points_mask_size = max(mask_limit)
print(d_points_on_surface.shape[0])
print(reduced_points_mask_size)

# create reduced array and copy using mask
d_reduced_points_on_surface = cuda.device_array((reduced_points_mask_size,4), dtype=np.float32)#reduced_points_mask_size
# threadsperblock = 1024 # max for rtx2070
# blockspergrid = (d_points_mask.shape[0] + (threadsperblock - 1)) // threadsperblock
# copy_to_reduced[blockspergrid, threadsperblock](d_points_mask, d_points_on_surface, d_reduced_points_on_surface)

# We still need a link to pixels. Create Matrix with ind to d_reduced_points_on_surface and create reduces version of d_sparseImg
d_image_to_index = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.int32)
threadsperblock = (32, 32) 
blockspergrid_x = math.ceil(d_image_to_index.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(d_image_to_index.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
initMatrix[blockspergrid, threadsperblock](d_image_to_index, -1) #init matrix on GPU
d_reducedSparseImg = cuda.device_array((reduced_points_mask_size,2), dtype=np.int32)
threadsperblock = 1024 # max for rtx2070
blockspergrid = (d_points_mask.shape[0] + (threadsperblock - 1)) // threadsperblock
create_reduced_arrays[blockspergrid, threadsperblock](d_points_mask, d_sparseImg, d_image_to_index, d_reducedSparseImg, d_points_on_surface, d_reduced_points_on_surface)


# construct a sparse matrix connecting pixels with nearby pixels
# search with a 6x6 2D Kernel (max possible distance for < 3px) on d_idx_matrix
# => result is 2D Array with Dim (reduced_points_mask_size, 36). So maximum possible = Buildplate Pixel * 36 * int32 = 1.3 GB. Should not crash card.
d_pixel_to_surface_points = cuda.device_array((d_reducedSparseImg.shape[0],25), np.int32)
threadsperblock = 128 # max for rtx2070
blockspergrid = (d_pixel_to_surface_points.shape[0] + (threadsperblock - 1)) // threadsperblock
gather_pixel_to_surface_points[blockspergrid, threadsperblock](d_pixel_to_surface_points, d_reducedSparseImg, d_image_to_index)
d_surface_point_to_pixels = cuda.device_array((d_reducedSparseImg.shape[0],36), np.int32)
threadsperblock = 128 # max for rtx2070
blockspergrid = (d_surface_point_to_pixels.shape[0] + (threadsperblock - 1)) // threadsperblock
gather_surface_point_to_pixels[blockspergrid, threadsperblock](d_surface_point_to_pixels, d_reduced_points_on_surface, d_image_to_index)


d_pixel_to_surface_points_distances = cuda.device_array((d_pixel_to_surface_points.shape[0],d_pixel_to_surface_points.shape[1]), np.float32)
threadsperblock = (32, 32) 
blockspergrid_x = math.ceil(d_pixel_to_surface_points_distances.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(d_pixel_to_surface_points_distances.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
initMatrix[blockspergrid, threadsperblock](d_pixel_to_surface_points_distances,0) #init matrix on GPU

threadsperblock = 128
blockspergrid = (d_pixel_to_surface_points_distances.shape[0] + (threadsperblock - 1)) // threadsperblock
calc_pixel_to_surface_points_distances[blockspergrid, threadsperblock](d_pixel_to_surface_points_distances, d_pixel_to_surface_points, d_reduced_points_on_surface, d_reducedSparseImg)


d_surface_point_to_pixels_distances = cuda.device_array((d_surface_point_to_pixels.shape[0],d_surface_point_to_pixels.shape[1]), np.float32)
threadsperblock = (32, 32) 
blockspergrid_x = math.ceil(d_surface_point_to_pixels_distances.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(d_surface_point_to_pixels_distances.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
initMatrix[blockspergrid, threadsperblock](d_surface_point_to_pixels_distances,0) #init matrix on GPU

threadsperblock = 128
blockspergrid = (d_surface_point_to_pixels_distances.shape[0] + (threadsperblock - 1)) // threadsperblock
calc_surface_point_to_pixels_distances[blockspergrid, threadsperblock](d_surface_point_to_pixels_distances, d_surface_point_to_pixels, d_reduced_points_on_surface, d_reducedSparseImg)


# init calculate exposure
#same for all arrays
d_sparse_pixel_exposure = cuda.device_array(int(d_reducedSparseImg.shape[0]), dtype=np.float32) 
threadsperblock = 128 # max for rtx2070
blockspergrid = (d_reducedSparseImg.shape[0] + (threadsperblock - 1)) // threadsperblock

d_sparse_surface_exposure = cuda.device_array(int(d_reducedSparseImg.shape[0]), dtype=np.float32) 
d_delta_goal = cuda.device_array(int(d_reducedSparseImg.shape[0]), dtype=np.float32)
d_steps = cuda.device_array(int(d_reducedSparseImg.shape[0]), dtype=np.float32) #np.asarray(range(d_reducedSparseImg.shape[0]), dtype=np.uint32)#

init_sparse_pixel_exposure[blockspergrid, threadsperblock](d_sparse_pixel_exposure, d_reduced_points_on_surface, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)

calc_surface_exposure[blockspergrid, threadsperblock](d_sparse_surface_exposure, d_sparse_pixel_exposure, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
result = np.zeros(1, dtype=np.float64)
threadsperblock = 128 # max for rtx2070
blockspergrid = (d_sparse_surface_exposure.shape[0] + (threadsperblock - 1)) // threadsperblock
get_max[blockspergrid,threadsperblock](result, d_sparse_surface_exposure)
scale_pixel_exposures[blockspergrid,threadsperblock](d_sparse_pixel_exposure,target_surface_exposure/result[0])
calc_surface_exposure[blockspergrid, threadsperblock](d_sparse_surface_exposure, d_sparse_pixel_exposure, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
calc_delta_goal[blockspergrid, threadsperblock](d_delta_goal, d_sparse_surface_exposure)
calc_next_step[blockspergrid, threadsperblock](d_steps, d_sparse_surface_exposure, d_sparse_pixel_exposure, d_pixel_to_surface_points, d_pixel_to_surface_points_distances)

# makeImage(d_sparse_pixel_exposure, d_reducedSparseImg, 'test_pixel.png')
# makeImage(d_sparse_surface_exposure, d_reducedSparseImg, 'test_surface.png')
# makeImage(d_steps, d_reducedSparseImg, 'test_steps.png')
# makeImage(d_delta_goal, d_reducedSparseImg, 'test_delta_goal.png', subtractMin=True)


# calculate exposure
result = np.zeros(1, dtype=np.float64)
for i in range(1000):
    calc_surface_exposure[blockspergrid, threadsperblock](d_sparse_surface_exposure, d_sparse_pixel_exposure, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
    calc_next_step[blockspergrid, threadsperblock](d_steps, d_sparse_surface_exposure, d_sparse_pixel_exposure, d_pixel_to_surface_points, d_pixel_to_surface_points_distances)
    step[blockspergrid, threadsperblock](d_sparse_pixel_exposure, d_steps)
    calc_surface_exposure[blockspergrid, threadsperblock](d_sparse_surface_exposure, d_sparse_pixel_exposure, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
    calc_delta_goal[blockspergrid, threadsperblock](d_delta_goal, d_sparse_surface_exposure)
    if (i%100 == 0):
        steps = d_steps.copy_to_host(stream=stream)
        surf_exp = d_sparse_surface_exposure.copy_to_host(stream=stream)
        pixel_exp = d_sparse_pixel_exposure.copy_to_host(stream=stream)
        stream.synchronize()
        print(np.mean(steps), np.min(steps), np.max(steps), '|', np.mean(pixel_exp), np.min(pixel_exp), np.max(pixel_exp),'|', np.mean(surf_exp), np.min(surf_exp), np.max(surf_exp))

# makeImage(d_sparse_pixel_exposure, d_reducedSparseImg, 'test_pixel.png')
# makeImage(d_sparse_surface_exposure, d_reducedSparseImg, 'test_surface.png')
# makeImage(d_steps, d_reducedSparseImg, 'test_steps.png')
# makeImage(d_delta_goal, d_reducedSparseImg, 'delta_goal.png', subtractMin=True)


threadsperblock = 128 #1024 # max for rtx2070
blockspergrid = (d_sparse_pixel_exposure.shape[0] + (threadsperblock - 1)) // threadsperblock
copy_exposure[blockspergrid, threadsperblock](d_out_img, d_sparse_pixel_exposure, d_reducedSparseImg)
out_img = d_out_img.copy_to_host(stream=stream)
stream.synchronize()
plt.imsave(fname='final_image.png', arr=out_img, cmap='gray_r', format='png')