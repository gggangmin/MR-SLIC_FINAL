# USAGE
# MR-SLIC_original
from pyspark.context import SparkContext
from skimage.segmentation import mark_boundaries
from skimage import io

import time
from skimage.segmentation import slic
from PIL import Image
from skimage.util import img_as_float

sc = SparkContext()

#number of partition
partition = 4

#resource file name
k=['map1','map2','map3','map4']


# loop over the number of segments
numSegments = 500




def Map(k):
    #time check
    start_time = time.time()
    
    
    #read imagez
    #submit image with job 
    img = Image.open( k +'.png')
    
    image = img_as_float(img)
    
    segments = slic(image, n_segments = numSegments, sigma = 5)
    
    finish_time = time.time()
    return start_time, finish_time
stime = time.time()
times = sc.parallelize(k,partition).map(Map).collect()
ftime = time.time()


import pandas as pd
data = pd.DataFrame(columns=['node','time'])
number = 0
node = 1
datat = pd.DataFrame(columns=['node','time'])
pd.options.display.float_format = '{:.6f}'.format
for i in times:
    datat.loc[number]=['node '+str(node),i[1]-i[0]]
    
    data.loc[number]=['node '+str(node)+' start',i[0]]
    number += 1
    data.loc[number]=['node '+str(node)+' finish',i[1]]
    number += 1
    node+=1

#network time
for i in [0,2,4,6]:
    print('network_ in_time : '+str(data['time'][i]-stime))
for i in [1,3,5,7]:
    print('network_ out_time : '+str(ftime-data['time'][i]))
    
    
data = data.sort_values(by=['time'], axis=0)

#slave time
print('slave node time')
print(data)

print(datat)

#whole time
print('whole_time : '+str(ftime-stime))


