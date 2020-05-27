# USAGE
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from PIL import Image
import os
import pandas as pd 

# loop over the number of segments

path = "/home/ubuntu/superpixel-benchmark/data/BSDS500/images/all"
file_list = os.listdir(path)
file_list_img = [file for file in file_list if file.endswith(".jpg")]

import time
time_dict = {}
for seg in [500,1000,1500,2000]:
    time_dict[str(seg)+'h'] = 0
    time_dict[str(seg)+'w'] = 0

for numSegments in [500,1000,1500,2000]:
    count = 0
    for img_name  in file_list_img:

        start_time = time.time()
        #read image
        img = Image.open(path+'/'+img_name)
        image = img_as_float(img)
        (img_w,img_h) = img.size

        img_shape = 'h'
        if img_h == 321:
            count +=1
            img_shape = 'w'


        segments = slic(image, n_segments = numSegments, sigma = 5)

        finish_time = time.time()

        if img_shape == 'h':
            time_dict[str(numSegments)+'h']+= finish_time-start_time
        else:
            time_dict[str(numSegments)+'w']+= finish_time-start_time
        
        
        '''
        di = '/home/ubuntu/exp/original_result/'+str(numSegments)
        
        if not os.path.exists(di):
            os.makedirs(di)
        df = pd.DataFrame(segments)
        df.to_csv(di+'/'+img_name[:-4]+".csv",header=None,index=None)
        '''

    time_dict[str(numSegments)+'h'] /= (500-count)
    time_dict[str(numSegments)+'w'] /= count
    print(str(numSegments)+'h - ',500-count)
    print(time_dict[str(numSegments)+'h'])
    print(str(numSegments)+'w - ',count)
    print(time_dict[str(numSegments)+'w'])
