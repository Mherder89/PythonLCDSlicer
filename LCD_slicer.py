# import os
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1" dont do this ... 

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
import gc
import time
import threading

cuda.select_device(0)

# constants
filedir = 'C:/Resin/articulated-dragon-mcgybeer20211115-748-6f8p9f/mcgybeer/articulated-dragon-mcgybeer/Dragon_v2/' #'C:/VSCode/PythonTest/' #
filename = 'Dragon_v2.stl' #'test.stl'# 'cube.stl'#
stop_loop = False
# end constants


from parallel_prefix_sum import *
from cuda_functions import *
from py_functions import *

mesh, voxel_bounds = initMesh(filedir + filename, scale = 0.5, scale_rel_to_buildplate = False, rotZ=0)
d_img = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.int32) # final image
d_img_1d = cuda.device_array((int(voxel_bounds[0]*voxel_bounds[1])), np.int32)
d_image_to_index = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.int32)

def run_loop(mesh, voxel_bounds, d_img, d_img_1d, d_image_to_index):
	global stop_loop
	for z_layer in range(voxel_bounds[2]):
 
		if (stop_loop):
			print('Loop aborted by user')
			break

		if (z_layer == 0): # first layer has height != zero
			continue

		# if (z_layer > 1):
		# 	continue

		print(z_layer, voxel_bounds[2]) # progress report

		# get a 2D svg vector path at z_layer
		slice_path = getPath(mesh, z_layer)
		# filledImg = slice_path.rasterize(pitch=1.0, origin=[0,0], resolution=voxel_bounds[0:2], fill=True, width=0)
		# plt.imsave(fname='output/debug_' + str(z_layer).zfill(4) + '.png', arr=filledImg, cmap='gray_r', format='png')
		# break

		# create a pixel image of sclice
		getDeviceSliceImg(d_img, slice_path)
		# saveImage(d_img, 'debug_' + str(z_layer).zfill(4) + '.png')
		# break

		# transform slice img to a sparse form mat[x,y] -> (x,y,1)
		d_sparseImg = convertToSparseForm(d_img, d_img_1d)
		# d_out_img = cuda.device_array((int(d_img.shape[0]),int(d_img.shape[1])), np.uint8)
		# sparseImg_to_img(d_out_img, d_sparseImg)
		# saveImage(d_out_img, 'debug_' + str(z_layer).zfill(4) + '.png')
		# break

		# get a thin part of the mesh. We need part of the mesh that is influenced by the exposure. Direction -z -> This is the part that has already been slices/exposed.
		d_faces, d_vertices = getThinSlice(mesh, z_layer)

		# find the closest point on the surface (only thin slice) for each pixel in d_sparseImg
		# filter by distance
		d_points_on_surface = getSurfacePoints(d_sparseImg, z_layer, d_faces, d_vertices)
		d_faces = None
		d_vertices = None
		gc.collect()

		# all points with large distance are infill and get full exposure
		markInfill(d_img, d_points_on_surface, d_sparseImg)
		# saveImage(d_img, 'img_infill_' + str(z_layer).zfill(4) + '.png')

		# create new list without infill points and points that where to close 

		# stream = cuda.stream()
		# sparseImg_dump = d_sparseImg.copy_to_host(stream=stream)
		# points_on_surface_dump = d_points_on_surface.copy_to_host(stream=stream)
		# image_to_index_dump = d_image_to_index.copy_to_host(stream=stream)
		# stream.synchronize()
		# try:
		d_reducedSparseImg, d_reduced_points_on_surface = reducePoints(d_sparseImg, d_points_on_surface, d_image_to_index)
		# except Exception as inst:
		# 	# dump relevant data to files
		# 	np.savetxt("output/sparseImg_dump.txt", sparseImg_dump, delimiter =", ")
		# 	np.savetxt("output/points_on_surface_dump.txt", points_on_surface_dump, delimiter =", ")
		# 	np.savetxt("output/image_to_index_dump.txt", image_to_index_dump, delimiter =", ")
		# 	print('Error in Layer ',z_layer)
		# 	print(type(inst))
		# 	break

		d_sparseImg = None
		d_points_on_surface = None
		gc.collect()

		d_out_img = cuda.device_array((int(d_img.shape[0]),int(d_img.shape[1])), np.uint8)
		sparseImg_to_img(d_out_img, d_reducedSparseImg)
		# saveImage(d_out_img, 'debug_' + str(z_layer).zfill(4) + '.png')
		# break

		# we need to know which surface points get light from which pixels and vice versa. For each connection we calculate the distance. This sets memory < recalculation.
		d_pixel_to_surface_points, d_pixel_to_surface_points_distances, d_surface_point_to_pixels, d_surface_point_to_pixels_distances = createSparseDependencies(d_reducedSparseImg, d_reduced_points_on_surface, d_image_to_index, z_layer)

		# find good exposure for pixels near surface. 
		d_sparse_pixel_exposure, d_sparse_surface_exposure, d_steps = initExposure(d_reducedSparseImg, d_reduced_points_on_surface, d_pixel_to_surface_points, d_pixel_to_surface_points_distances, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)

		# increase in small weighted steps until desired expusure for surface points is reached
		optimizeExposure(d_sparse_pixel_exposure, d_sparse_surface_exposure, d_steps, d_pixel_to_surface_points, d_pixel_to_surface_points_distances, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)

		# combine with infill
		copyExposureToFinalImg(d_img, d_sparse_pixel_exposure, d_reducedSparseImg)

		# export result
		saveImage(d_img, 'img_' + str(z_layer).zfill(4) + '.png')

		# collect garbage
		d_reducedSparseImg = None
		d_reduced_points_on_surface = None
		d_sparse_pixel_exposure = None
		d_steps = None
		d_sparse_surface_exposure = None
		d_pixel_to_surface_points = None
		d_pixel_to_surface_points_distances = None
		d_surface_point_to_pixels = None
		d_surface_point_to_pixels_distances = None
		gc.collect()
		# print(numba.core.runtime.rtsys.get_allocation_stats())
		meminfo = cuda.current_context().get_memory_info()
		print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		time.sleep(0.001)
	d_img = None
	d_img_1d = None
	d_image_to_index = None
	d_reducedSparseImg = None
	d_reduced_points_on_surface = None
	d_sparse_pixel_exposure = None
	d_steps = None
	d_sparse_surface_exposure = None
	d_pixel_to_surface_points = None
	d_pixel_to_surface_points_distances = None
	d_surface_point_to_pixels = None
	d_surface_point_to_pixels_distances = None
	gc.collect()

def get_input():
    global stop_loop
    keystrk=input('Press a key \n')
    # thread doesn't continue until key is pressed 
    print('You pressed: ', keystrk)
    stop_loop=True
    print('flag is now:', stop_loop)

n=threading.Thread(target=run_loop, args=[mesh, voxel_bounds, d_img, d_img_1d, d_image_to_index])
i=threading.Thread(target=get_input)
i.start()
n.start()