import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1" #dont do this ... 

import numpy as np
from numba import cuda
import numba
import math
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
cuda.select_device(0)

scan_max_threads = 1024 # number of threads per block the card can handle (select max possible value)
scan_array_size = scan_max_threads*2 # array size which can be accumulated parallel by one block
scan_nlevel = int(math.log2(scan_array_size)) # how many levels to sweep up and down with the parallel algorithm

@cuda.jit(device=True)
def exclusiv_parallel_prefix_sum(array, idx, nrows):
	#nrows = s_array.shape[0]
	#Up-Sweep - beware changes to nvidia link because idx already has factor 2 in it
	for d in range(scan_nlevel):
		two_pow_d = 2**d
		k = idx*two_pow_d - 1
		kidx = k + 2**(d+1)
		if (kidx < nrows): 
			array[kidx] += array[k + two_pow_d]
		cuda.syncthreads() # Wait until all threads have completed up sweep 
	#down sweep
	array[nrows-1] = 0
	for d in range(scan_nlevel-1,-1,-1):
		two_pow_d = 2**d
		k = idx*two_pow_d - 1
		kidx = k + 2**(d+1)
		if (kidx < nrows): 
			t = array[k + two_pow_d]
			array[k + two_pow_d] = array[kidx]
			array[kidx] +=  t
		cuda.syncthreads() # Wait until all threads have completed down sweep 

@cuda.jit(device=False)
def scan_preScan(array):
	threadID = cuda.threadIdx.x
	blockID = cuda.blockIdx.x # Block id in a 1D grid

	if threadID == 1 and blockID == 111:
		from pdb import set_trace; set_trace()

	idx_shared = threadID*2 # index in shared array 
	idx_device = idx_shared + blockID*scan_array_size # index in full array
	nrows = array.shape[0] - 1 # index limit for last block
	
	# Preload data into shared memory
	# s_array = cuda.shared.array(shape=(scan_array_size), dtype=array.dtype) 
	s_array = cuda.shared.array(shape=(scan_array_size), dtype=numba.int32) 
	if (idx_device < nrows): # each thread handles 2 fields
		s_array[idx_shared] = array[idx_device]
		s_array[idx_shared + 1] = array[idx_device + 1] 
	cuda.syncthreads()  # Wait until all threads finish preloading

	exclusiv_parallel_prefix_sum(s_array, idx_shared, scan_array_size) # scan s_array
	cuda.syncthreads()
	
	# output for exclusiv scan
	# if (idx2 < nrows):
	#   array[idx2] = s_array[idx]
	#   array[idx2 + 1] = s_array[idx + 1] 
	# output for inclusiv scan: switch the exclisiv scan to an inclusiv scan by shifting left and adding at the end
	if ((idx_shared + 2 == scan_array_size and idx_device + 1 < nrows) or idx_device + 1 == nrows): # last index in s_array or last index in array
		array[idx_device] = s_array[idx_shared + 1]
		array[idx_device + 1] += s_array[idx_shared + 1]
	elif (idx_device < nrows):
		array[idx_device] = s_array[idx_shared + 1]
		array[idx_device + 1] = s_array[idx_shared + 2] 


arr_dump = np.loadtxt("output/arr_dump.txt", dtype=np.int32, delimiter =", ")

stream = cuda.stream()
d_arr = cuda.to_device(np.asarray(arr_dump, dtype=np.int32), stream=stream) # 226182
stream.synchronize()

prescan_blockspergrid = (math.ceil(d_arr.shape[0]/2.0) + (scan_max_threads - 1)) // scan_max_threads #111
scan_preScan[prescan_blockspergrid, scan_max_threads](d_arr)
cuda.synchronize()

# try:
# 	prescan_blockspergrid = (math.ceil(d_arr.shape[0]/2.0) + (scan_max_threads - 1)) // scan_max_threads
# 	scan_preScan[prescan_blockspergrid, scan_max_threads](d_arr)
# 	cuda.synchronize()
# except Exception as inst:
# 	print('Error')
# 	print(type(inst))