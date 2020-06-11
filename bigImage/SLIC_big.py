# USAGE
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from PIL import Image
import os
import pandas as pd 

# loop over the number of segments

path = "/home/ubuntu/MR-SLIC_FINAL/bigImage/original"
file_list = os.listdir(path)
file_list_img = [file for file in file_list if file.endswith(".jpg")]

import time
time_dict = {}

for numSegments in [500,1000,1500,2000]:
    count = 0
    for img_name  in file_list_img:

        start_time = time.time()
        #read image
        img = Image.open(path+'/'+img_name)
        image = img_as_float(img)
        (img_w,img_h) = img.size



        segments = slic(image, n_segments = numSegments, sigma = 5)

        finish_time = time.time()

        time_dict[img_name[:-4]+':'+str(numSegments)] = finish_time-start_time
        
        

for k,v in time_dict.items():
    print(k)
    print(v)
