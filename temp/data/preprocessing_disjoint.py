# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:22:53 2020

@author: Gangmin
"""

# list for the number of image split
splits = [2,3,4,5,6,7,8]


import os
path = "C:/Users/Gangmin/Desktop/SLIC/논문작업/sourcecode/BSR/BSDS500/data/images/train"
file_list = os.listdir(path)
file_list_img = [file for file in file_list if file.endswith(".jpg")]

#save to csv file.
import pandas as pd
from PIL import Image

for split in splits:
    for img_name in file_list_img:
        # load the image and apply SLIC and extract (approximately)
        # the supplied number of segments
        img = Image.open(path +'/'+ img_name)
        size = img.size
        temp = size[1]//split
        remain = size[1]%split
        split_list =[]
        for i in range(split):
            split_list.insert(0,temp)
            if remain is not 0:
                split_list[0] = split_list[0]+1
                remain -=1
        index_start = 0
        index_end = 0
        for i in range(split):
            index_end += split_list[i]
            area = (0,index_start,size[0],index_end)
            cropped_img = img.crop(area)
            #cropped_img.show()
            
            #저장
            cropped_img.save("C:/Users/Gangmin/Desktop/SLIC/논문작업/sourcecode/cropped_img/"+str(split)+"/"+img_name[:-4]+"_"+str(i)+".jpg")
            
            index_start = index_end
            
        
    
    '''
    index = 0
    final_list = []
    for i in range(split):
        final_list.append(index)
        index += split_list[i]
    area = ()
    '''

    