import numpy as np
from numba import cuda
import math

#plot sparseArrays to output function def makeImage(sparse_array_to_plot, sparse_img_coords, fn, subtractMin = False):
@cuda.jit(device=False)
def initMatrix(out_matrix, val):
    idx, idy = cuda.grid(2)
    nrows = out_matrix.shape[0]
    ncolls = out_matrix.shape[1]
    if idx < nrows and idy < ncolls:
        out_matrix[idx, idy] = val

@cuda.jit
def get_max(result, values):
    idx = cuda.grid(1)
    nrows = values.shape[0]
    if idx < nrows:
        cuda.atomic.max(result, 0, values[idx])

@cuda.jit
def get_min(result, values):
    idx = cuda.grid(1)
    nrows = values.shape[0]
    if idx < nrows:
        cuda.atomic.min(result, 0, values[idx])

@cuda.jit(device=False)
def normTo255(out_array,array, min, max):
    idx = cuda.grid(1)
    nrows = array.shape[0]
    if idx < nrows:
        out_array[idx] = int((array[idx]-min)/max*255)#int((array[idx])/max*255)#

@cuda.jit(device=False)
def sparse_to_img(out_image, sparseArray, sparse_img_coords):
    idx = cuda.grid(1)
    nrows = sparseArray.shape[0]
    if idx < nrows:
        px, py = (sparse_img_coords[idx,0], sparse_img_coords[idx,1])
        out_image[px,py] = math.floor(sparseArray[idx])

@cuda.jit(device=False)
def matrix_mod2(out_matrix):
    idx, idy = cuda.grid(2)
    nrows = out_matrix.shape[0]
    ncolls = out_matrix.shape[1]
    if idx < nrows and idy < ncolls:
        out_matrix[idx, idy] = out_matrix[idx, idy] % 2

# matrix size must be grater than path size
@cuda.jit(device=False)
def count_crossings(out_matrix, path):
    idx, idy = cuda.grid(2)
    nrows = out_matrix.shape[0]
    ncolls = out_matrix.shape[1]

    # TODO use shared array
    # index = cuda.threadIdx.x + (cuda.threadIdx.x + 1)*cuda.threadIdx.y
    # path_len = path.shape[0]
    # s_path = cuda.shared.array(shape=(1024,2), dtype='float32')
    # loop_count = math.ceil(path_len / 1024.0)
    # for i in range(loop_count):
    #     shifted_index = index + i*1024
    #     if (shifted_index < path_len):
    #         s_path[index] = (path[shifted_index,0],path[shifted_index,1])
    #     cuda.syncthreads()  # Wait until all threads finish preloading

    if idx < nrows and idy < ncolls:
        # out_matrix[idx, idy] += 1
        path_size_1 = path.shape[0] - 1
        for i in range(path_size_1):
            i2 = i + 1
            # fast tests
            if (path[i, 0] <= idx and path[i2, 0] <= idx):
                continue
            if (path[i, 1] <= idy and path[i2, 1] <= idy):
                continue
            if (path[i, 1] >= idy and path[i2, 1] >= idy):
                continue
            delta_y = (path[i2, 1] - path[i, 1])
            if (abs(delta_y) > 1e-10):
                x_cross = (idy - path[i, 1]) * (path[i2, 0] - path[i, 0]) / delta_y + path[i, 0]
                if (x_cross >= idx): # crossed the line?
                    out_matrix[idx, idy] += 1

#CUDA slice to sparse
@cuda.jit(device=False)
def get_limit(some_array, out_limit):
    out_limit[0] = some_array[some_array.shape[0] - 1]

@cuda.jit(device=False)
def sliceImg_to_sparseImg(sliceImg, sparseImg, blocklength):
    idx = cuda.grid(1)
    nrows = sliceImg.shape[0]
    if (idx == 0 and sliceImg[0] == 1): # first case extra cause idx-1 in elif bellow
        sparseImg[0] = (0, 0)
    elif idx < nrows:
        if (sliceImg[idx-1] < sliceImg[idx]): # select only on step up
            sparse_pixel_number = sliceImg[idx] - 1 # shift by 1 cause inclusiv sum
            sparseImg[sparse_pixel_number] = (int(math.floor(idx / blocklength)), idx % blocklength)

@cuda.jit(device=False)
def sparsePixel_to_img(out_image, sparsePixels):
    idx = cuda.grid(1)
    nrows = sparsePixels.shape[0]
    if idx < nrows:
        px, py = (sparsePixels[idx,0], sparsePixels[idx,1])
        out_image[px,py] = 1


# CUDA sort faces so that the angle at C is always the obtuse angle (if on exists)
def sortObtuseAngleToC(faces,vertices):
    for i in range(faces.shape[0]):
        p1 = vertices[faces[i,0]]
        p2 = vertices[faces[i,1]]
        p3 = vertices[faces[i,2]]
        if (np.dot((p2-p1),(p3-p1)) < 0):
            faces[i] = np.roll(faces[i], 2)
        elif (np.dot((p1-p2),(p3-p2)) < 0):
            faces[i] = np.roll(faces[i], 1)

@cuda.jit(device=True, inline=True)
def norm3d(v1, len):
    return (v1[0]/len, v1[1]/len, v1[2]/len)

@cuda.jit(device=True, inline=True)
def length3d(v1):
    return math.sqrt(dot3d(v1,v1))

@cuda.jit(device=True, inline=True)
def dot3d(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@cuda.jit(device=True, inline=True)
def cross3d(v1, v2):
    return (v1[1]*v2[2] - v1[2]*v2[1], 
            v1[2]*v2[0] - v1[0]*v2[2], 
            v1[0]*v2[1] - v1[1]*v2[0])

@cuda.jit(device=True, inline=True)
def add3d(v1, v2):
    return (v1[0] + v2[0],
            v1[1] + v2[1],
            v1[2] + v2[2])

@cuda.jit(device=True, inline=True)
def scalar3d(a, v1):
    return (a*v1[0],
            a*v1[1],
            a*v1[2])

@cuda.jit(device=True, inline=True)
def diff3d(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

# Find the normal to the plane: n = (p2 - p1) x (p3 - p1)
@cuda.jit(device=True, inline=True)
def calcNormal3d(p1,p2,p3):
    return cross3d(diff3d(p2,p1),diff3d(p3,p1))

@cuda.jit(device=True, inline=True)
def calcClosestPoint(point, A, B, C):   #point and vertices p
    #point = (Pxy[0],Pxy[1],Pz*1.0)

    vCA = diff3d(C,A)
    vBA = diff3d(B,A)

    normal = cross3d(vCA,vBA) # Find the normal to the plane: n = (b - a) x (c - a);
    length = length3d(normal)                       # length of normal vector
    # if (length < 1e-2):
    #     return (-1000,-1000,-1000)
    normal = norm3d(normal,length)                  # Normalize normal vector CARE DIV 0 is degenerate
    planedist = dot3d(point,normal) - dot3d(A,normal)   # Project point onto the plane: p.dot(n) - a.dot(n)
    P = diff3d(point,scalar3d(planedist,normal))               # P = point - dist * n

    # Compute edge vector projection - p1
    vPA = diff3d(P, A)

    # Compute dot products
    dot_ca_ba = dot3d(vCA,vBA)
    dot_ba_ba = dot3d(vBA,vBA)
    dot_ca_ca = dot3d(vCA,vCA)
    dot_ba_pa = dot3d(vBA,vPA)
    dot_ca_pa = dot3d(vCA,vPA)
    
    denom = (dot_ca_ca * dot_ba_ba - dot_ca_ba * dot_ca_ba)

    invDenom = 1.0 / denom
    # https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    # Naming: a,b,c switch sign on sides (a,b,c) which are opposite sides of Points A,B,C
    c = (dot_ba_ba * dot_ca_pa - dot_ca_ba * dot_ba_pa) * invDenom
    b = (dot_ca_ca * dot_ba_pa - dot_ca_ba * dot_ca_pa) * invDenom
    a = 1 - c - b

    vCB = diff3d(C,B)
    dot_cb_ca = dot3d(vCB,vCA)

    vPB = diff3d(P,B)
    dot_cb_pb = dot3d(vCB, vPB)
    dot_cb_cb = dot3d(vCB,vCB)

    # return (P[0],P[1],P[2])

    # this structure depends on vertices to be ordered beforehand so that the angle at C is always the obtuse angle (if one exists)
    if (a < 0):
        if (b < 0):
            return (C[0],C[1],C[2]) # C ist nearest point
        elif (c < 0): 
            return (B[0],B[1],B[2]) # B ist nearest point
        else: # only a is negative -> p outside of extended BC line
            if (dot_cb_ca <= 0): # optuse at C
                if (dot_ca_pa <= 0):
                    return (A[0],A[1],A[2]) # A ist nearest point
                elif (dot_ca_pa < dot_ca_ca):
                    return add3d(A, scalar3d(dot_ca_pa/dot_ca_ca, vCA)) # projected point on AC-Line
                else:
                    if (dot_cb_pb >= dot_cb_cb):
                        return (C[0],C[1],C[2]) # C ist nearest point
                    elif (dot_cb_pb <= 0):
                        return (B[0],B[1],B[2]) # B ist nearest point
                    else:
                        return add3d(B, scalar3d(dot_cb_pb/dot_cb_cb, vCB)) # projected point on BC-Line
            else: # no optuse angles
                if (dot_cb_pb <= 0):
                    return (B[0],B[1],B[2]) # B ist nearest point
                elif(dot_cb_pb >= dot_cb_cb):
                    return (C[0],C[1],C[2]) # C ist nearest point
                else:
                    return add3d(B, scalar3d(dot_cb_pb/dot_cb_cb, vCB)) # projected point on BC-Line
    else:
        if (b < 0):
            if (c < 0):
                return (A[0],A[1],A[2]) # A ist nearest point
            else: # only b is negative
                if (dot_cb_ca <= 0): # optuse at C -> same procedure as only a negative and optuse at C
                    if (dot_ca_pa <= 0):
                        return (A[0],A[1],A[2]) # A ist nearest point
                    elif (dot_ca_pa < dot_ca_ca):
                        return add3d(A, scalar3d(dot_ca_pa/dot_ca_ca, vCA)) # projected point on AC-Line
                    else:
                        if (dot_cb_pb >= dot_cb_cb):
                            return (C[0],C[1],C[2]) # C ist nearest point
                        elif (dot_cb_pb <= 0):
                            return (B[0],B[1],B[2]) # B ist nearest point
                        else:
                            return add3d(B, scalar3d(dot_cb_pb/dot_cb_cb, vCB)) # projected point on BC-Line
                else: # no optuse angles
                    if (dot_ca_pa <= 0):
                        return (A[0],A[1],A[2]) # A ist nearest point
                    elif(dot_ca_pa >= dot_ca_ca):
                        return (C[0],C[1],C[2]) # C ist nearest point
                    else:
                        return add3d(A, scalar3d(dot_ca_pa/dot_ca_ca, vCA)) # projected point on AC-Line
        elif c < 0: # only c is negative
            if (dot_ba_pa <= 0): # optuse at p3
                return (A[0],A[1],A[2]) # A ist nearest point
            elif (dot_ba_pa >= dot_ba_ba):
                return (B[0],B[1],B[2]) # B ist nearest point
            else:
                return add3d(A, scalar3d(dot_ba_pa/dot_ba_ba, vBA)) # projected point on AB-Line
        else: # all three Barycentric coordinates are positive => #point is ON triangle
            return (P[0],P[1],P[2])

@cuda.jit(device=True, inline=True)
def pointDistance3DSq(p1,p2):
    delta = diff3d(p1,p2)
    return dot3d(delta,delta)

# returns true if point is irrelevent ie not in triangle bounds + offset
@cuda.jit(device=True, inline=True)
def fastTest(point, A, B, C, offset):
    xmin = (min(min(A[0],B[0]),C[0]) - offset) > point[0]
    xmax = (max(max(A[0],B[0]),C[0]) + offset) < point[0]
    ymin = (min(min(A[1],B[1]),C[1]) - offset) > point[1]
    ymax = (max(max(A[1],B[1]),C[1]) + offset) < point[1]
    return xmin or xmax or ymin or ymax


# CUDA find point on 3d mesh for very 2D point in sparseImg
# TODO put const (def at compile) somewhere top of file
min_distance_sq = 2*(0.5**2)
min_distance = np.sqrt(min_distance_sq)
max_distance = 2.6 # currently limited by 2d search square at about 3
max_distance_sq = max_distance**2
@cuda.jit(device=False, fastmath=True)
def findClosesPoints(out, sparseImg, z_layer, faces, vertices):
    # get all the data needed
    idx = cuda.grid(1)
    nrows = out.shape[0]
    if idx < nrows:            
        # out[idx,0], out[idx,1], out[idx,2], out[idx,3] = (200, -10,-10,-10)
        imgpoint = (sparseImg[idx,0]*1.0,sparseImg[idx,1]*1.0,z_layer*1.0)
        closestDistance_sq = max_distance_sq
        closestPoint = (-1,-1,-1)
        for face in faces:
            if (fastTest(imgpoint, vertices[face[0]], vertices[face[1]], vertices[face[2]],10)):
                continue
            #out[idx,0], out[idx,1], out[idx,2], out[idx,3] = (400, face[0],face[1],face[2])
            testpoint = calcClosestPoint(imgpoint, vertices[face[0]], vertices[face[1]], vertices[face[2]])
            distance_sq = pointDistance3DSq(testpoint, imgpoint)
            if (distance_sq < closestDistance_sq):
                if  (distance_sq < min_distance_sq):
                    closestDistance_sq = -1
                    closestPoint = (-2,-2,-2)
                    break # pixel touches mesh -> disregard of pixel
                else:
                    closestDistance_sq = distance_sq
                    closestPoint = (testpoint[0],testpoint[1],testpoint[2])
                # closestDistance_sq = distance_sq
                # closestPoint = (testpoint[0],testpoint[1],testpoint[2])
        #closestDistance_sq = (imgpoint[2] - closestPoint[2] )
        out[idx,0], out[idx,1], out[idx,2], out[idx,3] = (math.sqrt(closestDistance_sq), closestPoint[0], closestPoint[1], closestPoint[2])

#CUDA reduce d_points_on_surface: extract relevant points ie < max_distance
@cuda.jit(device=False)
def get_active_points(out_mask, d_points_on_surface):
    idx = cuda.grid(1)
    nrows = d_points_on_surface.shape[0]
    if idx < nrows:
        out_mask[idx] = d_points_on_surface[idx,1] >= 0

# numba cuda is missing cudaMemcpy, so we ned this (slow) workaround to get the last value in d_points_mask. Be aware of extra zero at end cause maybe it was uneven.
@cuda.jit(device=False)
def get_mask_limit(incl_scanned_array, d_mask_limit):
    d_mask_limit[0] = incl_scanned_array[incl_scanned_array.shape[0]-1]
    d_mask_limit[1] = incl_scanned_array[incl_scanned_array.shape[0]]

#since we can't mark the empty px with -1 (cause uint32) we need to use 0 and shift index by 1 in idx_matrix
@cuda.jit(device=False)
def create_reduced_arrays(incl_scanned_array_mask, sparseImg, image_to_index, reducedSparseImg, surface_points, reduced_surface_points):
    idx = cuda.grid(1)
    nrows = sparseImg.shape[0]
    # TODO check idx < incl_scanned_array_mask.shape[0]
    shifted_count = incl_scanned_array_mask[idx] - 1 # - 1 to get the index because of inclusiv scan
    if (idx == 0 and shifted_count == 0): # first case extra cause idx-1 in elif bellow
        image_to_index[sparseImg[idx,0], sparseImg[idx,1]] = 0
        reducedSparseImg[0] = (sparseImg[idx,0], sparseImg[idx,1])
        reduced_surface_points[0] = (surface_points[0,0], surface_points[0,1], surface_points[0,2], surface_points[0,3])
    elif idx < nrows:
        if (incl_scanned_array_mask[idx-1] < incl_scanned_array_mask[idx]): # select only on step up
            image_to_index[sparseImg[idx,0], sparseImg[idx,1]] = shifted_count 
            reducedSparseImg[shifted_count] = (sparseImg[idx,0], sparseImg[idx,1])
            reduced_surface_points[shifted_count] = (surface_points[idx,0], surface_points[idx,1], surface_points[idx,2], surface_points[idx,3])

# copy filling points to final image
@cuda.jit(device=False)
def mark_filling_points(sliceImg, points_on_surface, sparseImg):
    idx = cuda.grid(1)
    nrows = points_on_surface.shape[0]
    if idx < nrows:
        if (points_on_surface[idx,1] == -1):
            x = sparseImg[idx,0] 
            y = sparseImg[idx,1] 
            sliceImg[x,y] = 255


#CUDA gather_pixel_to_surface_points and gather_surface_point_to_pixels
#pixel influcneces which surface points?
# TODO use loops instead of hardcode to expand search radius
@cuda.jit(device=False)
def gather_pixel_to_surface_points(pixel_links, reducedSparseImg, image_to_index):
    idx = cuda.grid(1)
    nrows = pixel_links.shape[0]
    if idx < nrows:
        px = reducedSparseImg[idx, 0] #search around pixel
        py = reducedSparseImg[idx, 1]
        # TODO 25 is to small for when surface is above pixels! should be < 12x12 = 124 ... but how much? im afraid we need math ...
        # and how much memory should we sacrefice for some wierd edge cases?
        pixel_links[idx, 0] = image_to_index[px - 2, py - 2]
        pixel_links[idx, 1] = image_to_index[px - 2, py - 1]
        pixel_links[idx, 2] = image_to_index[px - 2, py]
        pixel_links[idx, 3] = image_to_index[px - 2, py + 1]
        pixel_links[idx, 4] = image_to_index[px - 2, py + 2]
        pixel_links[idx, 5] = image_to_index[px - 1, py - 2]
        pixel_links[idx, 6] = image_to_index[px - 1, py - 1]
        pixel_links[idx, 7] = image_to_index[px - 1, py]
        pixel_links[idx, 8] = image_to_index[px - 1, py + 1]
        pixel_links[idx, 9] = image_to_index[px - 1, py + 2]
        pixel_links[idx, 10] = image_to_index[px, py - 2]
        pixel_links[idx, 11] = image_to_index[px, py - 1]
        pixel_links[idx, 12] = image_to_index[px, py]
        pixel_links[idx, 13] = image_to_index[px, py + 1]
        pixel_links[idx, 14] = image_to_index[px, py + 2]
        pixel_links[idx, 15] = image_to_index[px + 1, py - 2]
        pixel_links[idx, 16] = image_to_index[px + 1, py - 1]
        pixel_links[idx, 17] = image_to_index[px + 1, py]
        pixel_links[idx, 18] = image_to_index[px + 1, py + 1]
        pixel_links[idx, 19] = image_to_index[px + 1, py + 2]
        pixel_links[idx, 20] = image_to_index[px + 2, py - 2]
        pixel_links[idx, 21] = image_to_index[px + 2, py - 1]
        pixel_links[idx, 22] = image_to_index[px + 2, py]
        pixel_links[idx, 23] = image_to_index[px + 2, py + 1]
        pixel_links[idx, 24] = image_to_index[px + 2, py + 2]

# surface points is influenced by which pixels?
@cuda.jit(device=False)
def gather_surface_point_to_pixels(pixel_links, reduced_points_on_surface, image_to_index):
    idx = cuda.grid(1)
    nrows = pixel_links.shape[0]
    if idx < nrows:
        px = int(math.floor(reduced_points_on_surface[idx, 1])) #search around surface point projected to image
        py = int(math.floor(reduced_points_on_surface[idx, 2]))
        #TODO automate hardcode
        pixel_links[idx, 0] = image_to_index[px - 2, py - 2]
        pixel_links[idx, 1] = image_to_index[px - 2, py - 1]
        pixel_links[idx, 2] = image_to_index[px - 2, py]
        pixel_links[idx, 3] = image_to_index[px - 2, py + 1]
        pixel_links[idx, 4] = image_to_index[px - 2, py + 2]
        pixel_links[idx, 5] = image_to_index[px - 2, py + 3]
        pixel_links[idx, 6] = image_to_index[px - 1, py - 2]
        pixel_links[idx, 7] = image_to_index[px - 1, py - 1]
        pixel_links[idx, 8] = image_to_index[px - 1, py]
        pixel_links[idx, 9] = image_to_index[px - 1, py + 1]
        pixel_links[idx, 10] = image_to_index[px - 1, py + 2]
        pixel_links[idx, 11] = image_to_index[px - 1, py + 3]
        pixel_links[idx, 12] = image_to_index[px, py - 2]
        pixel_links[idx, 13] = image_to_index[px, py - 1]
        pixel_links[idx, 14] = image_to_index[px, py]
        pixel_links[idx, 15] = image_to_index[px, py + 1]
        pixel_links[idx, 16] = image_to_index[px, py + 2]
        pixel_links[idx, 17] = image_to_index[px, py + 3]
        pixel_links[idx, 18] = image_to_index[px + 1, py - 2]
        pixel_links[idx, 19] = image_to_index[px + 1, py - 1]
        pixel_links[idx, 20] = image_to_index[px + 1, py]
        pixel_links[idx, 21] = image_to_index[px + 1, py + 1]
        pixel_links[idx, 22] = image_to_index[px + 1, py + 2]
        pixel_links[idx, 23] = image_to_index[px + 1, py + 3]
        pixel_links[idx, 24] = image_to_index[px + 2, py - 2]
        pixel_links[idx, 25] = image_to_index[px + 2, py - 1]
        pixel_links[idx, 26] = image_to_index[px + 2, py]
        pixel_links[idx, 27] = image_to_index[px + 2, py + 1]
        pixel_links[idx, 28] = image_to_index[px + 2, py + 2]
        pixel_links[idx, 29] = image_to_index[px + 2, py + 3]
        pixel_links[idx, 30] = image_to_index[px + 3, py - 2]
        pixel_links[idx, 31] = image_to_index[px + 3, py - 1]
        pixel_links[idx, 32] = image_to_index[px + 3, py]
        pixel_links[idx, 33] = image_to_index[px + 3, py + 1]
        pixel_links[idx, 34] = image_to_index[px + 3, py + 2]
        pixel_links[idx, 35] = image_to_index[px + 3, py + 3]


# d_sparse_pixel_neighbours_distances
@cuda.jit(device=False)
def calc_pixel_to_surface_points_distances(distances, pixel_links, reduced_points_on_surface, reducedSparseImg, z_layer):
    idx = cuda.grid(1)
    nrows = distances.shape[0]
    if idx < nrows:
        point_img = (reducedSparseImg[idx,0], reducedSparseImg[idx,1], z_layer)
        for i in range(25):
            distances[idx,i] = -1
            surface_point_idx = pixel_links[idx,i]
            if (surface_point_idx >= 0):
                point_surface = (reduced_points_on_surface[surface_point_idx,1], reduced_points_on_surface[surface_point_idx,2], reduced_points_on_surface[surface_point_idx,3])
                distance = math.sqrt(pointDistance3DSq(point_surface, point_img))
                if (distance > min_distance and distance < max_distance):
                    distances[idx,i] = distance
                else:
                    pixel_links[idx,i] = -1 # remove link if distance is out of bounds
                    distances[idx,i] = 0


# d_sparse_relevant_pixels_distances
@cuda.jit(device=False)
def calc_surface_point_to_pixels_distances(distances, pixel_links, reduced_points_on_surface, reducedSparseImg, z_layer):
    idx = cuda.grid(1)
    nrows = distances.shape[0]
    if idx < nrows:
        for i in range(36):
            point_surface = (reduced_points_on_surface[idx,1], reduced_points_on_surface[idx,2], reduced_points_on_surface[idx,3])
            pixel_idx = pixel_links[idx,i]
            if (pixel_idx >= 0):
                point_img = (reducedSparseImg[pixel_idx,0], reducedSparseImg[pixel_idx,1], z_layer)
                distance = math.sqrt(pointDistance3DSq(point_surface, point_img))
                if (distance > min_distance and distance < max_distance):
                    distances[idx,i] = distance
                else:
                    pixel_links[idx,i] = -1 # remove link if distance is out of bounds
                    distances[idx,i] = 0


#CUDA calculate exposure
target_surface_exposure = 500 # goal intesity for the surface. Should barely be solid.
# model of the light intesity decay at a distance from the pixel
@cuda.jit(device=True)
def distance_to_decay_factor(dist):
    # linear interpolation of lookup table
    d1 = min_distance # 0.5
    d2 = (max_distance_sq - min_distance) / 2.0 #1.75
    d3 = max_distance_sq #3.0

    f1 = 1.0
    f2 = 0.2
    f3 = 0.0

    if (dist < d1):
        return f1
    elif (dist < d2):
        return (f2-f1)/(d2-d1) * (dist - d1) + f1
    elif (dist < d3):
        return (f3-f2)/(d3-d2) *  (dist - d2) + f2
    else:
        return 0.0



@cuda.jit(device=False)
def init_sparse_pixel_exposure(exposure, points_on_surface, surface_point_to_pixels, surface_point_to_pixels_distances):
    idx = cuda.grid(1)
    nrows = exposure.shape[0]
    if idx < nrows:
        weight = 0
        for i in range(36):
            idx2 = surface_point_to_pixels[idx,i]
            if (idx2 >= 0):
                weight += distance_to_decay_factor(surface_point_to_pixels_distances[idx, i])#1.0/surface_point_to_pixels_distances[idx, i]
        if (weight < 0.01):
            weight = 0.01
        # (1-dist)  -> favor pixels further from surface
        # weight    -> decrease inner curves by weight with number of influences
        decay_f = distance_to_decay_factor(points_on_surface[idx,0])
        if (decay_f) == 0:
            exposure[idx] = 255
        else:
            exposure[idx] = (1.0 - distance_to_decay_factor(points_on_surface[idx,0])) / weight 

@cuda.jit(device=False)
def calc_surface_exposure(surface_exposure, pixel_exposures, surface_point_to_pixels, surface_point_to_pixels_distances):
    idx = cuda.grid(1)
    nrows = surface_exposure.shape[0]
    if idx < nrows:
        out = 0
        for i in range(36):
            idx2 = surface_point_to_pixels[idx,i]
            if (idx2 >= 0):
                out += distance_to_decay_factor(surface_point_to_pixels_distances[idx, i]) * pixel_exposures[idx2]
        surface_exposure[idx] = out

@cuda.jit(device=False)
def calc_next_step(steps, sparse_surface_exposure, pixel_to_surface_points, pixel_to_surface_points_distances):
    idx = cuda.grid(1)
    nrows = steps.shape[0]
    if idx < nrows:
        grad = 0
        for i in range(25):
            idx2 = pixel_to_surface_points[idx,i]
            if idx2 >= 0:
                # grad for delta**2
                delta_sq_ist = (target_surface_exposure - sparse_surface_exposure[idx2])**2
                delta_sq_plus1 = (target_surface_exposure - (sparse_surface_exposure[idx2] + distance_to_decay_factor(pixel_to_surface_points_distances[idx, i])))**2
                grad += delta_sq_ist - delta_sq_plus1
        steps[idx] = grad 

@cuda.jit(device=False)
def calc_delta_goal(delta_out, sparse_surface_exposure):
    idx = cuda.grid(1)
    nrows = delta_out.shape[0]
    if idx < nrows:
        delta_out[idx] = abs(target_surface_exposure - sparse_surface_exposure[idx])

@cuda.jit(device=False)
def calc_next_step(steps, surface_exposure, pixel_exposures, pixel_to_surface_points, pixel_to_surface_points_distances):
    idx = cuda.grid(1)
    nrows = steps.shape[0]
    if idx < nrows:        
        mean_factor = 0.0
        weightsum = 0.0
        bFreeze = False # if any target surface point if overexposed -> stop
        for i in range(25):
            if (bFreeze):
                continue
            idx2 = pixel_to_surface_points[idx,i]
            if idx2 >= 0:
                surface_exp = surface_exposure[idx2]
                weight = distance_to_decay_factor(pixel_to_surface_points_distances[idx, i])
                weightsum += weight
                delta = target_surface_exposure - surface_exp
                mean_factor += delta / surface_exp * weight
                bFreeze = (delta < 0)
        if (bFreeze):
            steps[idx] = 0
        else:
            if (weightsum < 0.001):
                weightsum = 0.001
            steps[idx] = 0.1 * (pixel_exposures[idx] * (mean_factor/weightsum) * (abs((target_surface_exposure - surface_exposure[idx]) / target_surface_exposure)**2))

@cuda.jit(device=False)
def step(pixel_exposures, steps):
    idx = cuda.grid(1)
    nrows = pixel_exposures.shape[0]
    if idx < nrows:
        if (steps[idx] > 0):
            pixel_exposures[idx] += steps[idx]
        if (pixel_exposures[idx] > 255):
            pixel_exposures[idx] = 255
        elif (pixel_exposures[idx] < 0):
            pixel_exposures[idx] = 0

@cuda.jit(device=False)
def scale_pixel_exposures(pixel_exposures, factor):
    idx = cuda.grid(1)
    nrows = pixel_exposures.shape[0]
    if idx < nrows:
        pixel_exposures[idx] = min(255, pixel_exposures[idx] * factor) 


# add exposure to d_out_img
@cuda.jit(device=False)
def copy_exposure(out_image, sparse_pixel_exposure, reducedSparseImg):
    idx = cuda.grid(1)
    nrows = sparse_pixel_exposure.shape[0]
    if idx < nrows:
        x = reducedSparseImg[idx,0] 
        y = reducedSparseImg[idx,1] 
        out_image[x,y] = sparse_pixel_exposure[idx]