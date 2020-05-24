#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX
from cpython cimport bool

import numpy as np
cimport numpy as cnp

from ..util import regular_grid
#처리해야 할 것 일렬 array를 받아서 index대로 분리해서 사용하고(연속으로 읽을 수 있음), [] 씌워주기, index 모양 수
def _slic_cythonM(double[:,::1] distance, Py_ssize_t[:,::1] nearest_segments, Py_ssize_t[:,::1] index, long[::1] dimension, Py_ssize_t[::1] listi):
    cdef Py_ssize_t depth, height, width
    depth = dimension[0]
    height = dimension[1]
    width = dimension[2]
    cdef Py_ssize_t[:, :, ::1] nearest_segmentsO \
        = np.empty((depth, height, width), dtype=np.intp)
    cdef double[:, :, ::1] distanceO \
        = np.empty((depth, height, width), dtype=np.double)
    cdef Py_ssize_t i,z,y,x,z_min, y_min, x_min, temp
    distanceO[:, :, :] = DBL_MAX
    cdef Py_ssize_t n_node = index[0].shape[0]
   
    #인덱스 생성 
    cdef Py_ssize_t[::1] startpixel = np.empty(3, dtype=np.intp) 
    
    for i in range(n_node):
        index[0][i] = listi[index[0][i]-1]
    
           
            
    #새로운 코드
    with nogil:
        temp = 0   
        for i in range(0,n_node):
            z_min = 0
            y_min = index[0][i]
             
            
            for z in range(z_min,z_min+1):
                for y in range(y_min,y_min+index[1][i]):
                    for x in range(0, width):
                        if distanceO[z,y,x] > distance[temp+y-y_min,x]:
                            distanceO[z,y,x] = distance[temp+y-y_min,x]
                            nearest_segmentsO[z,y,x] = nearest_segments[temp+y-y_min,x] + nearest_segmentsO[z,y_min,x]
            temp += index[1][i]
    return np.asarray(nearest_segmentsO)
