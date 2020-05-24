# USAGE
# MR-SLIC_original
from pyspark.context import SparkContext
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from PIL import Image
sc = SparkContext()

#number of partition
partition = 4

#resource file name
k=['house1','house2','house3','house4']



# loop over the number of segments
numSegments = 500

import time
start_time = time.time()
#read image
img = Image.open('/home/wjdrmf314/MR-SLIC_NEW/resource/' + 'house' +'.jpg')
image = img_as_float(img)

segments = slic(image, n_segments = numSegments, sigma = 5)

finish_time = time.time()
print(finish_time-start_time)
