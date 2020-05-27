# import the necessary packages
from skimage.segmentation.slic_superpixels import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import sys
import os


# list for the number of image split
splits = [8]


path = "/home/ubuntu/superpixel-benchmark/data/BSDS500/images/all"
file_list = os.listdir(path)
file_list_img = [file for file in file_list if file.endswith(".jpg")]

#save to csv file.
import pandas as pd


check = True
# loop over the number of segments
for numSegments in [500,1000,1500,2000]:
    for img_name in file_list_img:

        # load the image and apply SLIC and extract (approximately)
        # the supplied number of segments
        img = Image.open(path +'/'+ img_name)
        image = img_as_float(img)
        segments = slic(image,n_segments=numSegments,sigma=5,max_iter=1)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        size =img.size
        img_shape = "h"
        if size[0]>size[1]:
            img_shape = "w"
        init_list = []
        dist_list = []
        for i,v in enumerate(segments):
            if v[0] not in init_list:
                init_list.append(v[0])
                dist_list.append(i)
        #ilist 생성
        partition = 8
        height = len(init_list)
        temp = height//partition
        remain = height%partition
        split_list =[]
        for i in range(partition):
            split_list.insert(0,temp)
            if remain is not 0:
                split_list[0] = split_list[0]+1
                remain -=1

        split_list.sort(reverse=True)


        iteration = 0
        dist_list.append(size[1])

        while True:

            if iteration>max(split_list):
                break


            index = 0
            for e,i in enumerate(split_list):
                overlap = min(iteration,i)
                first = dist_list[max(0,index-overlap)]
                index += i
                last = dist_list[min(index+overlap,len(init_list))]
                area = (0,first,size[0],last)


                cropped_img = img.crop(area)
                save = "/home/ubuntu/superpixel-benchmark/data/BSDS500/grid/"+img_shape+"/"+str(numSegments)+"/"+str(iteration)+"/"+img_name[:-4]+'_'+str(e)+".jpg"
                dir =  "/home/ubuntu/superpixel-benchmark/data/BSDS500/grid/"+img_shape+"/"+str(numSegments)+"/"+str(iteration)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                cropped_img.save(save)

            iteration += 1

