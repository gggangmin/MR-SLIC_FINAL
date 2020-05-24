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
def _slic_cythonM(double[:,::1] dist_array, Py_ssize_t[:,::1] seg_array, \
                  Py_ssize_t[::1] pi_list,Py_ssize_t[::1] split_list, \
                  long[::1] dimension, Py_ssize_t[::1] sp_acc, \
                  Py_ssize_t[:,::1] indlist,Py_ssize_t[::1] listi, \
                  Py_ssize_t[::1] sp_list,Py_ssize_t overlap, Py_ssize_t seg_w):
    cdef Py_ssize_t depth, height, width
    depth = dimension[0]
    height = dimension[1]
    width = dimension[2]
    cdef Py_ssize_t[:, :, ::1] nearest_segmentsO \
        = np.empty((depth, height, width), dtype=np.intp)
    cdef double[:, :, ::1] distanceO \
        = np.empty((depth, height, width), dtype=np.double)
    cdef Py_ssize_t i,z,y,x,o,z_min, y_min, x_min, temp,t,b_index,start,last,extension,p,ptemp
    distanceO[:, :, :] = DBL_MAX
    cdef Py_ssize_t n_node = indlist[0].shape[0]
    cdef Py_ssize_t[::1] boundary\
        = np.empty((seg_w*2), dtype=np.intp)
    #인덱스 생성 
    
    
    
    
    
    for i in range(n_node):
        indlist[0][i] = listi[indlist[0][i]]
    #overlap이 0이 아닌경우만 slic_master가 실행됨
    
    
    with nogil:
        temp = 0
        for i in range(0,n_node):
            z_min = 0
            y_min = indlist[0][i]
            for z in range(z_min,z_min+1):
                start = 0
                last = height
                if i!=0 :
                    start = y_min-pi_list[sp_acc[i]-1]
                if i!= n_node-1:
                    last = y_min+split_list[i]+pi_list[sp_acc[i+1]]

                extension = 0
                for o in range(0,overlap):
                    if o !=0: 
                        if o>sp_list[i]-1:
                            break
                        extension += pi_list[sp_acc[i]-o-1]
                for y in range(start,last):                             
                    for x in range(0, width):
                        if i == 0:
                            distanceO[z,y,x] = dist_array[y,x]
                            nearest_segmentsO[z,y,x] = seg_array[y,x]*10
                        else:
                            ptemp=0
                            for p in range(0,seg_w*2):
                                if boundary[p] == nearest_segmentsO[z,y,x]:
                                    ptemp = 1
                            #if (y>=start and y<y_min+pi_list[sp_acc[i]]) and np.in1d(boundary,nearest_segmentsO[z,y,x]):
                            if (y>=start and y<y_min+pi_list[sp_acc[i]]) and ptemp ==1 :
                                pass
                                #nearest_segmentsO[z,y,x] = 999
                            else:
                                distanceO[z,y,x] = dist_array[temp+y-start+extension,x]
                                nearest_segmentsO[z,y,x] = seg_array[temp+y-start+extension,x]*10+i

                        if   i!= n_node-1 and y == y_min+split_list[i]-pi_list[sp_acc[i+1]-1]:
                            if x ==0:
                                #초기화
                                for t in range(seg_w*2):
                                    boundary[t] = -1
                                b_index=0
                                boundary[b_index] = nearest_segmentsO[z,y,x]
                            else:
                                if boundary[b_index-1] != nearest_segmentsO[z,y,x]:
                                    boundary[b_index] = nearest_segmentsO[z,y,x]
                                    b_index +=1
                            #boundary = np.append(boundary,nearest_segmentsO[z,y,x])


            temp += indlist[1][i]
    return np.asarray(nearest_segmentsO)
