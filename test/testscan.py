from parallel_prefix_sum import *

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
import gc

cuda.select_device(0)

# 56 blockpergrid 113430 arraysize
# Error on pre_scan der 56 blocks

rng = np.random.default_rng(1337)
von = 4
bis = scan_array_size*scan_array_size*8
for i in range (1000):
	cuda.current_context().deallocations.clear()
	randomint = rng.integers(low=von/2, high=bis/2,size=1)[0]*2
	testarray = np.asarray(rng.integers(low=0, high=1,size=randomint), dtype=np.int32)
	d_testarray = cuda.to_device(testarray)
	scan_array(d_testarray)
	python_scan = np.cumsum(testarray)[-1]
	if (d_testarray[-1] != python_scan):
		print(i, randomint, d_testarray[-1])
		d_testarray = None
		gc.collect()
		break
	d_testarray = None
	gc.collect()




#test array scan
# test with ones
# rng = np.random.default_rng(6584)
# von = 4
# bis = scan_array_size*scan_array_size*8
# for i in range (100):
#     cuda.current_context().deallocations.clear()
#     randomint = rng.integers(low=von/2, high=bis/2,size=1)[0]*2
#     testarray = np.asarray(np.ones(randomint), dtype=np.int32)
#     d_testarray = cuda.to_device(testarray)
#     scan_array(d_testarray)
#     if (d_testarray[-1] != randomint):
#         print(i, randomint, d_testarray[-1])
#         break
#test with random numbers 0-1
# rng = np.random.default_rng(6584)
# von = 4
# bis = scan_array_size*scan_array_size*8
# for i in range (10):
#     cuda.current_context().deallocations.clear()
#     randomint = rng.integers(low=von/2, high=bis/2,size=1)[0]*2
#     testarray = np.asarray(rng.integers(low=0, high=1,size=randomint), dtype=np.int32)
#     d_testarray = cuda.to_device(testarray)
#     scan_array(d_testarray)
#     python_scan = np.cumsum(testarray)[-1]
#     if (d_testarray[-1] != python_scan):
#         print(i, randomint, d_testarray[-1])
#         break
# #test with random numbers 0-255
# rng = np.random.default_rng(6584)
# von = scan_array_size
# bis = scan_array_size*scan_array_size
# for i in range (10):
#     cuda.current_context().deallocations.clear()
#     randomint = rng.integers(low=von/2, high=bis/2,size=1)[0]*2
#     testarray = np.asarray(rng.integers(low=0, high=255,size=randomint), dtype=np.int32)
#     d_testarray = cuda.to_device(testarray)
#     scan_array(d_testarray)
#     python_scan = np.cumsum(testarray)[-1]
#     if (d_testarray[-1] != python_scan):
#         print(i, randomint, d_testarray[-1])
#         break

