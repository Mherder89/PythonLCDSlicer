import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
import gc
cuda.select_device(0)

from parallel_prefix_sum import *


points_mask_dump = np.loadtxt("output/array_dump.txt", dtype=np.uint32, delimiter =", ")

stream = cuda.stream()
d_points_mask = cuda.to_device(np.asarray(points_mask_dump, dtype=np.uint32), stream=stream)
stream.synchronize()

try:
	scan_array(d_points_mask)
	cuda.synchronize()
except Exception as inst:
	print('Error')
	print(type(inst))