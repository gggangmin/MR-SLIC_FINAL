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

path = "/home/ubuntu/MR-SLIC_FINAL/bigImage/original"
file_list = os.listdir(path)
file_list_img = [file for file in file_list if file.endswith(".jpg")]

#save to csv file.
import pandas as pd
from PIL import Image

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
        
        init_list = []
        dist_list = []
        for i,v in enumerate(segments):
            if v[0] not in init_list:
                init_list.append(v[0])
                dist_list.append(i)
        #ilist 생성
        for partition in [2,3,4,5,6,7,8]:
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


            dist_list.append(size[1])            

            index = 0
            for e,i in enumerate(split_list):
                first = dist_list[index]
                index += i
                last = dist_list[index]
                area = (0,first,size[0],last)

                cropped_img = img.crop(area)
                di =  "/home/ubuntu/MR-SLIC_FINAL/bigImage/disjoint/"+str(numSegments)+"/"+str(partition)
                save = "/home/ubuntu/MR-SLIC_FINAL/bigImage/disjoint/"+str(numSegments)+"/"+str(partition)+"/"+img_name[:-4]+"_"+str(e)+".jpg"
                if not os.path.exists(di):
                    os.makedirs(di)
                cropped_img.save(save)

