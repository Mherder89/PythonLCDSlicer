import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
import gc
cuda.select_device(0)

# constants
filedir = 'C:/Resin/articulated-dragon-mcgybeer20211115-748-6f8p9f/mcgybeer/articulated-dragon-mcgybeer/Dragon_v2/' #'C:/VSCode/PythonTest/' #
filename = 'Dragon_v2.stl' #'test.stl'# 'cube.stl'#
# end constants


from parallel_prefix_sum import *
from cuda_functions import *
from py_functions import *

mesh, voxel_bounds = initMesh(filedir + filename, scale_rel_to_buildplate = 0.4)
d_img = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.int32) # final image
d_img_1d = cuda.device_array((int(voxel_bounds[0]*voxel_bounds[1])), np.int32)
d_image_to_index = cuda.device_array((int(voxel_bounds[0]),int(voxel_bounds[1])), np.int32)

z_layer = 70

slice_path = getSlice(mesh, z_layer)
getDeviceSliceImg(d_img, slice_path)
d_sparseImg = convertToSparseForm(d_img, d_img_1d)
d_faces, d_vertices = getThinSlice(mesh, z_layer)
d_points_on_surface = getSurfacePoints(d_sparseImg, z_layer, d_faces, d_vertices)
d_faces = None
d_vertices = None
gc.collect()
markInfill(d_img, d_points_on_surface, d_sparseImg)

d_sparseImg = None
d_points_on_surface = None
d_image_to_index = None
gc.collect()

sparseImg_dump = np.loadtxt("output/sparseImg_dump.txt", dtype=np.int32, delimiter =", ")
points_on_surface_dump = np.loadtxt("output/points_on_surface_dump.txt", dtype=np.float32, delimiter =", ")
image_to_index_dump = np.loadtxt("output/image_to_index_dump.txt", dtype=np.int32, delimiter =", ")

stream = cuda.stream()
d_sparseImg = cuda.to_device(np.asarray(sparseImg_dump, dtype=np.int32), stream=stream)
d_points_on_surface = cuda.to_device(np.asarray(points_on_surface_dump, dtype=np.float32), stream=stream)
d_image_to_index = cuda.to_device(np.asarray(image_to_index_dump, dtype=np.int32), stream=stream)
stream.synchronize()

try:
	d_reducedSparseImg, d_reduced_points_on_surface = reducePoints(d_sparseImg, d_points_on_surface, d_image_to_index)
except Exception as inst:
	print('Error in Layer ',z_layer)
	print(type(inst))