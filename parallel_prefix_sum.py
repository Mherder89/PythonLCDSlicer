import numpy as np
from numba import cuda
import math

#TODO make this work for arbitrary many levels (recursiv structure) and test with small scan_max_threads
#TODO clean and optimize

# CUDA parallel_prefix_sum (scan) with numba CUDA. 
# https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
# like numpy.cumsum but for each entry. In Mathematica it is called Accumulate[].
# It is needed to pick values from a large array into a smaler array
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

    idx_shared = threadID*2 # index in shared array 
    idx_device = idx_shared + blockID*scan_array_size # index in full array
    nrows = array.shape[0] - 1 # index limit for last block
    
    # Preload data into shared memory
    s_array = cuda.shared.array(shape=(scan_array_size), dtype=array.dtype) 
    if (idx_device < nrows): # each thread handles 2 fields
        s_array[idx_shared] = array[idx_device]
        s_array[idx_shared + 1] = array[idx_device + 1] 
    cuda.syncthreads()  # Wait until all threads finish preloading

    exclusiv_parallel_prefix_sum(s_array, idx_shared, scan_array_size) # scan s_array
    
    # output for exclusiv scan
    # if (idx2 < nrows):
    #   array[idx2] = s_array[idx]
    #   array[idx2 + 1] = s_array[idx + 1] 
    # output for inclusiv scan: switch the exclisiv scan to an inclusiv scan by shifting left and adding at the end
    if (idx_shared + 2 == scan_array_size or idx_device + 1 == nrows): # last index in s_array or last index in array
        array[idx_device] = s_array[idx_shared + 1]
        array[idx_device + 1] += s_array[idx_shared + 1]
    elif (idx_device < nrows):
        array[idx_device] = s_array[idx_shared + 1]
        array[idx_device + 1] = s_array[idx_shared + 2] 

@cuda.jit(device=False)
def scan_addTwoBlocks(array):
    threadID = cuda.threadIdx.x*2 + scan_array_size
    nrows = array.shape[0]
    if (threadID < nrows):
        array[threadID] += array[scan_array_size-1]
        array[threadID+1] += array[scan_array_size-1]

@cuda.jit(device=False)
def scan_addNBlocksOnDevice(array, nsum):
    threadID = cuda.threadIdx.x*2
    threadID2 = threadID + 1
    
    s_array = cuda.shared.array(shape=(scan_array_size), dtype=array.dtype) # create and copy data to block-sum-array
    cuda.syncthreads() # needed?
    
    pp1 = (threadID + 1)*scan_array_size - 1
    pp2 = (threadID2 + 1)*scan_array_size - 1
    if (pp1 < array.shape[0]):
        s_array[threadID] = array[pp1] # copy end of blocks from pre-scan
    if (pp2 < array.shape[0]):
        s_array[threadID2] = array[pp2] # copy end of blocks from pre-scan
    cuda.syncthreads()

    exclusiv_parallel_prefix_sum(s_array, threadID, nsum + nsum%2) # scan s_array, need to be even

    # add sums to d_array. Is one block fast or better to copy s_array to multiple blocks?
    for i in range(1,nsum): #first block need no addition
        pp1=threadID*nsum
        if (pp1 < array.shape[0]):
            array[pp1] += s_array[i] # copy end of blocks from pre-scan
        pp2=threadID2*nsum
        if (pp2 < array.shape[0]):
            array[pp2] += s_array[i] # copy end of blocks from pre-scan
    
    # nrows = d_block_sum.shape[0]
    # if (d_block_sum < nrows):
    #     d_block_sum[threadID] = d_array[threadID*(array_size-1)]

@cuda.jit(device=False)
def scan_copyBlockEnds(array, block_sum):
    idx = cuda.grid(1)
    idx2 = (idx + 1)*scan_array_size - 1

    idx_limit = block_sum.shape[0]
    idx2_limit = array.shape[0]
    if (idx < idx_limit):
        if (idx2 < idx2_limit): # exlude the extra to make it even
            block_sum[idx] = array[idx2]
        else:
            block_sum[idx] = array[idx2_limit - 1] # copy last element

@cuda.jit(device=False)
def scan_addBlockEnds(array, block_sum):
    idx = cuda.grid(1)
    idx2 = int(math.floor(cuda.blockIdx.x/2)) - 1 # add sum from previous block
    idx_limit = array.shape[0]
    idx2_limit = block_sum.shape[0]
    if (idx < idx_limit and 0 <= idx2 and idx2 < idx2_limit):
        array[idx] += block_sum[idx2]


def scan_array(arr):
    if (arr.shape[0] % 2 == 1):
        print("array not even!") # TODO: maybe copy and add 0?
        return

    prescan_blockspergrid = (math.ceil(arr.shape[0]/2.0) + (scan_max_threads - 1)) // scan_max_threads
    scan_preScan[prescan_blockspergrid, scan_max_threads](arr) # scans blocks of length max_threads*2

    # 1 Block -> allready done
    if prescan_blockspergrid == 1:
        pass
    elif (prescan_blockspergrid == 2): # 2 Blocks -> just add end of block 1 to second block
        scan_addTwoBlocks[1, scan_max_threads](arr)
    elif (prescan_blockspergrid <= scan_array_size): #write end of block directly to shared array and scan
        d_block_sum = cuda.device_array(prescan_blockspergrid + prescan_blockspergrid%2, np.uint32) # needs to be even
        threadsperblock = min(scan_max_threads, d_block_sum.shape[0])
        blockspergrid = (d_block_sum.shape[0] + (threadsperblock - 1)) // threadsperblock
        scan_copyBlockEnds[blockspergrid, threadsperblock](arr, d_block_sum) # copy end of all blocks
        scan_preScan[1, scan_max_threads](d_block_sum) # only one block so prescan = full scan
        threadsperblock = scan_max_threads
        blockspergrid = (arr.shape[0] + (threadsperblock - 1)) // threadsperblock
        scan_addBlockEnds[blockspergrid, threadsperblock](arr, d_block_sum)
    else: # d_block_sum doesnt fit in array_size, expand one level, currently no need to go further.
        # print("prescan_blockspergrid > array_size!")
        threadsperblock = scan_max_threads
        d_block_sum = cuda.device_array(prescan_blockspergrid + prescan_blockspergrid%2, np.uint32) # needs to be even
        blockspergrid = (d_block_sum.shape[0] + (threadsperblock - 1)) // threadsperblock
        scan_copyBlockEnds[blockspergrid, threadsperblock](arr, d_block_sum) # copy end of all blocks
        threadsperblock = scan_max_threads #1024 # max for rtx2070
        sum_prescan_blockspergrid = (math.ceil(d_block_sum.shape[0]/2.0) + (scan_max_threads - 1)) // scan_max_threads
        scan_preScan[sum_prescan_blockspergrid, threadsperblock](d_block_sum) # only one block so prescan = full scan
        if (sum_prescan_blockspergrid == 2): #cant be one. TODO Merge with  elif (prescan_blockspergrid <= scan_array_size):
            scan_addTwoBlocks[1, scan_max_threads](d_block_sum)
        else:
            #block block sum
            threadsperblock = scan_max_threads
            d_block_block_sum = cuda.device_array(sum_prescan_blockspergrid + sum_prescan_blockspergrid%2, np.uint32) # needs to be even
            blockspergrid = (d_block_block_sum.shape[0] + (threadsperblock - 1)) // threadsperblock
            scan_copyBlockEnds[blockspergrid, threadsperblock](d_block_sum, d_block_block_sum) # copy end of all blocks
            threadsperblock = scan_max_threads
            blockspergrid = 1
            scan_preScan[blockspergrid, threadsperblock](d_block_block_sum) # only one block so prescan = full scan
            threadsperblock = scan_max_threads
            blockspergrid = (d_block_sum.shape[0] + (threadsperblock - 1)) // threadsperblock
            scan_addBlockEnds[blockspergrid, threadsperblock](d_block_sum, d_block_block_sum)
        threadsperblock = scan_max_threads
        blockspergrid = (arr.shape[0] + (threadsperblock - 1)) // threadsperblock
        scan_addBlockEnds[blockspergrid, threadsperblock](arr, d_block_sum)