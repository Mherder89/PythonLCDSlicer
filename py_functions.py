import trimesh
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt

from cuda_functions import *
from parallel_prefix_sum import *

#load mesh
def initMesh(fpath, rotZ = 0, scale = 1, scale_rel_to_buildplate = False):

	mesh_raw = trimesh.load_mesh(fpath)
	mesh_raw.visual.face_colors = [100, 100, 100, 255] # give it a color
	#mesh_raw.show(viewer='gl') # and show it
	# transform mesh
	origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
	mesh_out = mesh_raw.copy()

	# transform, bounds = trimesh.bounds.oriented_bounds(mesh_out, angle_digits=2, ordered=True, normal=zaxis)
	# print(bounds)
	# mesh_out.apply_transform(transform)
	# mesh_out.apply_transform(trimesh.transformations.translation_matrix(bounds/2.0))
	# mesh_out.apply_transform(trimesh.transformations.reflection_matrix([0,0,0],[-1,0,1]))

	rotZ *= math.pi/180
	Rz = trimesh.transformations.rotation_matrix(rotZ, zaxis)
	mesh_out.apply_transform(Rz)

	mesh_out.rezero()
	bounds = [int(np.ceil(mesh_out.bounding_box.bounds[1,0])), int(np.ceil(mesh_out.bounding_box.bounds[1,1])), int(np.ceil(mesh_out.bounding_box.bounds[1,2]))]
	if (scale_rel_to_buildplate):
		scale *= min([3840.0/bounds[0],2400.0/bounds[1],4000.0/bounds[2]])
	else:
		scale *= 1.0/0.05
	mesh_out.apply_transform(trimesh.transformations.scale_matrix(scale, origin))

	# mesh_out.rezero()
	# print(voxel_bounds)
	# print(mesh.bounding_box.bounds)
	bounds = [int(np.ceil(mesh_out.bounding_box.bounds[1,0])), int(np.ceil(mesh_out.bounding_box.bounds[1,1])), int(np.ceil(mesh_out.bounding_box.bounds[1,2]))]
	print('bounds [px]: ', bounds)
	print('bounds [mm]: ', np.asarray(bounds, dtype=np.float64) * 0.05)
	bounds[0] += 5 # 2px left 3px right border
	bounds[1] += 5 
	bounds[0] += bounds[0] % 2
	bounds[1] += bounds[1] % 2
	# shift xy by 2 pixels for border and +0.5 , so pixel index coordinate is mid pixel
	mesh_out.apply_transform(trimesh.transformations.translation_matrix([2.5, 2.5, 0])) # shift xy by half a pixel (+2px Boder), so pixel index coordinate is mid pixel
	return mesh_out, bounds

def getPath(mesh, z_layer):
	#cut layer
	project_z0_matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,-z_layer],[0,0,0,1]]
	slice = mesh.section(plane_origin=[0,0,z_layer], plane_normal=[0, 0, 1])
	slice_2D, to_3D = slice.to_planar(to_2D=project_z0_matrix, normal=[0, 0, 1], check=False)
	# from PIL import ImageChops
	# # borderIMG = slice_2D.rasterize(pitch=1.0, origin=[0,0], resolution=voxel_bounds[0:2], fill=False, width=0)
	# # binaryImg = ImageChops.difference(filledImg,borderIMG) 
	# # sadly this does not work, Pixels that are too close are discarded later but take up resources :-(
	# # need slicer for only inner Pixels, no touching!
	# binaryImg = filledImg
	# print(slice_2D.plot_entities())
	# sparseImg = np.nonzero(binaryImg)
	# sparseImg = np.transpose([sparseImg[1],sparseImg[0]])
	return slice_2D

def getDeviceSliceImg(img, slice_2DPath):
	threadsperblock = (32, 32) 
	blockspergrid_x = math.ceil(img.shape[0] / threadsperblock[0])
	blockspergrid_y = math.ceil(img.shape[1] / threadsperblock[1])
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	initMatrix[blockspergrid, threadsperblock](img, 0)
	cuda.synchronize()
	for i in range(slice_2DPath.polygons_closed.shape[0]):
		if slice_2DPath.polygons_closed[i] is not None:
			h_path = np.asarray(list(slice_2DPath.polygons_closed[i].exterior.coords), dtype=np.float64)
			d_path = cuda.to_device(h_path)
			count_crossings[blockspergrid, threadsperblock](img, d_path)
			cuda.synchronize()
			d_path = None

	matrix_mod2[blockspergrid, threadsperblock](img)
	cuda.synchronize()

def convertToSparseForm(img2d, img1d):
	# not needed?
	# threadsperblock = 128
	# blockspergrid = (img1d.shape[0] + (threadsperblock - 1)) // threadsperblock
	# initArray[blockspergrid, threadsperblock](img1d, 0)
	threadsperblock = 128
	blockspergrid = (img1d.shape[0] + (threadsperblock - 1)) // threadsperblock
	copy_reshape_img[blockspergrid, threadsperblock](img1d, img2d)
	cuda.synchronize()
	scan_array(img1d)
	cuda.synchronize()
	d_sliceImg_scan_limit = cuda.device_array(1, img1d.dtype)
	get_limit[1, 1](img1d, d_sliceImg_scan_limit)
	cuda.synchronize()
	slice_img_scan_limit = int(d_sliceImg_scan_limit.copy_to_host()[0])
	# print(slice_img_scan_limit)
	d_sliceImg_scan_limit = None
	d_sparseImg = cuda.device_array((slice_img_scan_limit,2), dtype=np.int32)
	threadsperblock = 128
	blockspergrid = (img1d.shape[0] + (threadsperblock - 1)) // threadsperblock
	img_to_sparseImg[blockspergrid, threadsperblock](img1d, d_sparseImg, img2d.shape[1])
	cuda.synchronize()
	return d_sparseImg

def getThinSlice(mesh, z_layer):
	# get closest point on faces
	# copy data to device
	sliceme = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[0, 0, -1], plane_origin=[0,0,z_layer])
	buttomcut = z_layer-3
	if (buttomcut < 0.5):
		buttomcut = 0.5
	sliceme = trimesh.intersections.slice_mesh_plane(sliceme, plane_normal=[0, 0, 1], plane_origin=[0,0,buttomcut])
	#sliceme.bounding_box.bounds
	# print(sliceme.faces.shape)
	sliceme.remove_degenerate_faces(height=1e-03) # remove and face with one side < height. < 1e-4 still has errors
	# print(sliceme.faces.shape)
	#sliceme.show(viewer='gl', flags={'wireframe': True})
	faces = np.asarray(sliceme.faces, dtype=np.int32)#np.asarray(error_faces, dtype=np.uint32) #
	vertices = np.asarray(sliceme.vertices, dtype=np.float32)
	sortObtuseAngleToC(faces,vertices)
	# d_sparseImg = cuda.to_device(np.asarray(sparseImg, dtype=np.int32), stream=stream) # list of pixels inside (or at border) of the mesh
	stream = cuda.stream()
	d_faces = cuda.to_device(faces, stream=stream) # faces of the cut mesh. Each is a index reference to 3 vertices.
	d_vertices = cuda.to_device(vertices, stream=stream) # vertices of the cut mesh. Lowest z coord has to be zIndex.
	stream.synchronize()
	return d_faces, d_vertices

def getSurfacePoints(d_sparseImg, z, d_faces, d_vertices):
	#calc closest points
	d_points_on_surface = cuda.device_array((d_sparseImg.shape[0],4), np.float32) # output: array of closes points on any face for input point in sparseImg
	threadsperblock = 128
	blockspergrid = (d_points_on_surface.shape[0] + (threadsperblock - 1)) // threadsperblock
	findClosesPoints[blockspergrid, threadsperblock](d_points_on_surface, d_sparseImg, z, d_faces, d_vertices)
	cuda.synchronize()
	# points_on_surface = d_points_on_surface.copy_to_host(stream=stream)
	# stream.synchronize()
	# points_on_surface.shape[0]
	return d_points_on_surface

def markInfill(d_img, d_points_on_surface, d_sparseImg):
	threadsperblock = (32, 32) 
	blockspergrid_x = math.ceil(d_img.shape[0] / threadsperblock[0])
	blockspergrid_y = math.ceil(d_img.shape[1] / threadsperblock[1])
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	initMatrix[blockspergrid, threadsperblock](d_img,0)
	cuda.synchronize()
	threadsperblock = 128
	blockspergrid = (d_points_on_surface.shape[0] + (threadsperblock - 1)) // threadsperblock
	mark_filling_points[blockspergrid, threadsperblock](d_img, d_points_on_surface, d_sparseImg)
	cuda.synchronize()
	# out_img = d_img.copy_to_host(stream=stream)
	# stream.synchronize()
	# plt.imsave(fname='filling_pixels.png', arr=out_img, cmap='gray_r', format='png')

def reducePoints(d_sparseImg, d_points_on_surface, d_image_to_index):

	# stream = cuda.stream()

	# reduce d_points_on_surface: extract relevant points ie < max_distance
	# generate a binary mask for relevant points
	d_points_mask = cuda.device_array(int(d_points_on_surface.shape[0]+d_points_on_surface.shape[0]%2), dtype=np.int32) # scan algo needs even size
	threadsperblock = 128
	blockspergrid = (d_points_mask.shape[0] + (threadsperblock - 1)) // threadsperblock
	initArray[blockspergrid, threadsperblock](d_points_mask, 0)
	cuda.synchronize()
	get_active_points[blockspergrid, threadsperblock](d_points_mask, d_points_on_surface)
	cuda.synchronize()

	#scan mask array and extract largest value as reduced array size
	# stream = cuda.stream()
	# points_mask_dump = d_points_mask.copy_to_host(stream=stream)
	# stream.synchronize()
	# try:
	scan_array(d_points_mask)
	cuda.synchronize()
	# except Exception as inst:
	# 	# dump relevant data to files
	# 	np.savetxt("output/array_dump.txt", points_mask_dump, delimiter =", ")
	# 	print('Error with array scan!')
	# 	print(type(inst))
	# 	return

	d_mask_limit = cuda.device_array(2, np.int32)
	get_mask_limit[1, 1](d_points_mask, d_mask_limit)
	cuda.synchronize()
	mask_limit = d_mask_limit.copy_to_host()
	# mask_limit = d_mask_limit.copy_to_host(stream=stream)
	# stream.synchronize()
	reduced_points_mask_size = max(mask_limit)
	# print(d_points_on_surface.shape[0])
	# print(reduced_points_mask_size)

	# create reduced array and copy using mask
	d_reduced_points_on_surface = cuda.device_array((reduced_points_mask_size,4), dtype=np.float32)#reduced_points_mask_size
	# d_mask_limit = None

	# We still need a link to pixels. Create Matrix with ind to d_reduced_points_on_surface and create reduces version of d_sparseImg
	threadsperblock = (32, 32) 
	blockspergrid_x = math.ceil(d_image_to_index.shape[0] / threadsperblock[0])
	blockspergrid_y = math.ceil(d_image_to_index.shape[1] / threadsperblock[1])
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	initMatrix[blockspergrid, threadsperblock](d_image_to_index, -1) #init matrix on GPU
	cuda.synchronize()
	d_reducedSparseImg = cuda.device_array((reduced_points_mask_size,2), dtype=np.int32)
	threadsperblock = 128
	blockspergrid = (d_points_mask.shape[0] + (threadsperblock - 1)) // threadsperblock
	create_reduced_arrays[blockspergrid, threadsperblock](d_points_mask, d_sparseImg, d_image_to_index, d_reducedSparseImg, d_points_on_surface, d_reduced_points_on_surface)
	cuda.synchronize()
	# d_points_mask = Nones
	return d_reducedSparseImg, d_reduced_points_on_surface

def createSparseDependencies(d_reducedSparseImg, d_reduced_points_on_surface, d_image_to_index, z_layer):
	# construct a sparse matrix connecting pixels with nearby pixels
	# search with a 6x6 2D Kernel (max possible distance for < 3px) on d_idx_matrix
	# => result is 2D Array with Dim (reduced_points_mask_size, 36). So maximum possible = Buildplate Pixel * 36 * int32 = 1.3 GB. Should not crash card.
	d_pixel_to_surface_points = cuda.device_array((d_reducedSparseImg.shape[0],25), np.int32)
	threadsperblock = 128
	blockspergrid = (d_pixel_to_surface_points.shape[0] + (threadsperblock - 1)) // threadsperblock
	gather_pixel_to_surface_points[blockspergrid, threadsperblock](d_pixel_to_surface_points, d_reducedSparseImg, d_image_to_index)
	cuda.synchronize()
	d_surface_point_to_pixels = cuda.device_array((d_reducedSparseImg.shape[0],36), np.int32)
	threadsperblock = 128
	blockspergrid = (d_surface_point_to_pixels.shape[0] + (threadsperblock - 1)) // threadsperblock
	gather_surface_point_to_pixels[blockspergrid, threadsperblock](d_surface_point_to_pixels, d_reduced_points_on_surface, d_image_to_index)
	cuda.synchronize()

	d_pixel_to_surface_points_distances = cuda.device_array((d_pixel_to_surface_points.shape[0],d_pixel_to_surface_points.shape[1]), np.float32)
	threadsperblock = (32, 32) 
	blockspergrid_x = math.ceil(d_pixel_to_surface_points_distances.shape[0] / threadsperblock[0])
	blockspergrid_y = math.ceil(d_pixel_to_surface_points_distances.shape[1] / threadsperblock[1])
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	initMatrix[blockspergrid, threadsperblock](d_pixel_to_surface_points_distances,0) #init matrix on GPU
	cuda.synchronize()

	threadsperblock = 128
	blockspergrid = (d_pixel_to_surface_points_distances.shape[0] + (threadsperblock - 1)) // threadsperblock
	calc_pixel_to_surface_points_distances[blockspergrid, threadsperblock](d_pixel_to_surface_points_distances
																			, d_pixel_to_surface_points
																			, d_reduced_points_on_surface
																			, d_reducedSparseImg
																			, z_layer)
	cuda.synchronize()

	d_surface_point_to_pixels_distances = cuda.device_array((d_surface_point_to_pixels.shape[0],d_surface_point_to_pixels.shape[1]), np.float32)
	threadsperblock = (32, 32) 
	blockspergrid_x = math.ceil(d_surface_point_to_pixels_distances.shape[0] / threadsperblock[0])
	blockspergrid_y = math.ceil(d_surface_point_to_pixels_distances.shape[1] / threadsperblock[1])
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	initMatrix[blockspergrid, threadsperblock](d_surface_point_to_pixels_distances,0) #init matrix on GPU
	cuda.synchronize()

	threadsperblock = 128
	blockspergrid = (d_surface_point_to_pixels_distances.shape[0] + (threadsperblock - 1)) // threadsperblock
	calc_surface_point_to_pixels_distances[blockspergrid, threadsperblock](d_surface_point_to_pixels_distances
																			, d_surface_point_to_pixels
																			, d_reduced_points_on_surface
																			, d_reducedSparseImg
																			, z_layer)
	cuda.synchronize()

	return d_pixel_to_surface_points, d_pixel_to_surface_points_distances, d_surface_point_to_pixels, d_surface_point_to_pixels_distances

def initExposure(d_reducedSparseImg, d_reduced_points_on_surface, d_pixel_to_surface_points, d_pixel_to_surface_points_distances, d_surface_point_to_pixels, d_surface_point_to_pixels_distances):
	# init calculate exposure
	#same for all arrays
	d_sparse_pixel_exposure = cuda.device_array(int(d_reducedSparseImg.shape[0]), dtype=np.float32) 
	threadsperblock = 128
	blockspergrid = (d_reducedSparseImg.shape[0] + (threadsperblock - 1)) // threadsperblock

	d_sparse_surface_exposure = cuda.device_array(int(d_reducedSparseImg.shape[0]), dtype=np.float32) 
	d_steps = cuda.device_array(int(d_reducedSparseImg.shape[0]), dtype=np.float32) #np.asarray(range(d_reducedSparseImg.shape[0]), dtype=np.uint32)#

	init_sparse_pixel_exposure[blockspergrid, threadsperblock](d_sparse_pixel_exposure, d_reduced_points_on_surface, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
	cuda.synchronize()

	calc_surface_exposure[blockspergrid, threadsperblock](d_sparse_surface_exposure, d_sparse_pixel_exposure, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
	cuda.synchronize()

	result = np.zeros(1, dtype=np.float32)
	cuda.device_array(int(d_reducedSparseImg.shape[0]), dtype=np.float32) 
	threadsperblock = 128
	blockspergrid = (d_sparse_surface_exposure.shape[0] + (threadsperblock - 1)) // threadsperblock
	get_max[blockspergrid,threadsperblock](result, d_sparse_surface_exposure)
	cuda.synchronize()

	scale_pixel_exposures[blockspergrid,threadsperblock](d_sparse_pixel_exposure,target_surface_exposure/result[0])
	cuda.synchronize()

	calc_surface_exposure[blockspergrid, threadsperblock](d_sparse_surface_exposure, d_sparse_pixel_exposure, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
	cuda.synchronize()

	calc_next_step[blockspergrid, threadsperblock](d_steps, d_sparse_surface_exposure, d_sparse_pixel_exposure, d_pixel_to_surface_points, d_pixel_to_surface_points_distances)
	cuda.synchronize()

	# d_out_img_test = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.uint8) # final slicer image
	# makeImage(d_out_img_test, d_sparse_pixel_exposure, d_reducedSparseImg, 'test_pixel.png')
	# makeImage(d_out_img_test, d_sparse_surface_exposure, d_reducedSparseImg, 'test_surface.png')
	# makeImage(d_out_img_test, d_steps, d_reducedSparseImg, 'test_steps.png')
	# makeImage(d_out_img_test, d_delta_goal, d_reducedSparseImg, 'test_delta_goal.png', subtractMin=True)

	return d_sparse_pixel_exposure, d_sparse_surface_exposure, d_steps

def optimizeExposure(d_sparse_pixel_exposure, d_sparse_surface_exposure, d_steps, d_pixel_to_surface_points, d_pixel_to_surface_points_distances, d_surface_point_to_pixels, d_surface_point_to_pixels_distances):
	threadsperblock = 128
	blockspergrid = (d_sparse_surface_exposure.shape[0] + (threadsperblock - 1)) // threadsperblock
	# calculate exposure
	# result = np.zeros(1, dtype=np.float64)
	for i in range(100):
		calc_surface_exposure[blockspergrid, threadsperblock](d_sparse_surface_exposure, d_sparse_pixel_exposure, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
		cuda.synchronize()
		calc_next_step[blockspergrid, threadsperblock](d_steps, d_sparse_surface_exposure, d_sparse_pixel_exposure, d_pixel_to_surface_points, d_pixel_to_surface_points_distances)
		cuda.synchronize()
		step[blockspergrid, threadsperblock](d_sparse_pixel_exposure, d_steps)
		cuda.synchronize()
		calc_surface_exposure[blockspergrid, threadsperblock](d_sparse_surface_exposure, d_sparse_pixel_exposure, d_surface_point_to_pixels, d_surface_point_to_pixels_distances)
		cuda.synchronize()
		# if (i%100 == 0):
		# 	steps = d_steps.copy_to_host()
		# 	surf_exp = d_sparse_surface_exposure.copy_to_host()
		# 	pixel_exp = d_sparse_pixel_exposure.copy_to_host()
		# 	print(np.mean(steps), np.min(steps), np.max(steps), '|', np.mean(pixel_exp), np.min(pixel_exp), np.max(pixel_exp),'|', np.mean(surf_exp), np.min(surf_exp), np.max(surf_exp))

	# makeImage(d_out_img_test, d_sparse_pixel_exposure, d_reducedSparseImg, 'test_pixel.png')
	# makeImage(d_out_img_test, d_sparse_surface_exposure, d_reducedSparseImg, 'test_surface.png')
	# makeImage(d_out_img_test, d_steps, d_reducedSparseImg, 'test_steps.png')
	# makeImage(d_out_img_test, d_delta_goal, d_reducedSparseImg, 'delta_goal.png', subtractMin=True)

def copyExposureToFinalImg(d_out_img, d_sparse_pixel_exposure, d_reducedSparseImg):
	threadsperblock = 128
	blockspergrid = (d_sparse_pixel_exposure.shape[0] + (threadsperblock - 1)) // threadsperblock
	copy_exposure[blockspergrid, threadsperblock](d_out_img, d_sparse_pixel_exposure, d_reducedSparseImg)
	cuda.synchronize()

def saveImage(d_out_img, fn):
	out_img = d_out_img.copy_to_host()
	plt.imsave(fname='output/' + fn, arr=out_img.transpose(), cmap='gray_r', format='png')

def sparseImg_to_img(out_img, sparseImg):
	threadsperblock = (32, 32) 
	blockspergrid_x = math.ceil(out_img.shape[0] / threadsperblock[0])
	blockspergrid_y = math.ceil(out_img.shape[1] / threadsperblock[1])
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	initMatrix[blockspergrid, threadsperblock](out_img,0) #init matrix on GPU
	cuda.synchronize()
	threadsperblock = 128
	blockspergrid = (sparseImg.shape[0] + (threadsperblock - 1)) // threadsperblock
	sparsePixel_to_img[blockspergrid, threadsperblock](out_img, sparseImg)
	cuda.synchronize()

def makeImage(d_out_img, sparse_array_to_plot, sparse_img_coords, fn, subtractMin = False):
	# d_out_img = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.uint8) # final slicer image
	d_sparse_scaled = cuda.device_array(int(sparse_array_to_plot.shape[0]), dtype=np.float32) 

	threadsperblock = (32, 32) 
	blockspergrid_x = math.ceil(d_out_img.shape[0] / threadsperblock[0])
	blockspergrid_y = math.ceil(d_out_img.shape[1] / threadsperblock[1])
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	initMatrix[blockspergrid, threadsperblock](d_out_img,0) #init matrix on GPU
	cuda.synchronize()

	result = np.zeros(1, dtype=np.float64)
	threadsperblock = 128
	blockspergrid = (sparse_array_to_plot.shape[0] + (threadsperblock - 1)) // threadsperblock
	get_max[blockspergrid,threadsperblock](result, sparse_array_to_plot)
	array_max = result[0]
	get_min[blockspergrid,threadsperblock](result, sparse_array_to_plot)
	array_min = result[0]
	print(array_min, array_max)
	
	threadsperblock = 128
	blockspergrid = (d_sparse_scaled.shape[0] + (threadsperblock - 1)) // threadsperblock
	if (subtractMin):
		normTo255[blockspergrid, threadsperblock](d_sparse_scaled, sparse_array_to_plot, array_min, array_max)
	else:
		normTo255[blockspergrid, threadsperblock](d_sparse_scaled, sparse_array_to_plot, 0, array_max)
	cuda.synchronize()

	threadsperblock = 128
	blockspergrid = (sparse_array_to_plot.shape[0] + (threadsperblock - 1)) // threadsperblock
	sparse_to_img[blockspergrid, threadsperblock](d_out_img, d_sparse_scaled, sparse_img_coords)
	cuda.synchronize()
	d_sparse_scaled = None
	out_img = d_out_img.copy_to_host()
	plt.imsave(fname=fn, arr=out_img.transpose(), cmap='gray_r', format='png')