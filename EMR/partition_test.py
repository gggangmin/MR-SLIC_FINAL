
# coding: utf-8

# In[18]:

# USAGE
# MR-SLIC_original
from pyspark.context import SparkContext
from skimage.segmentation import mark_boundaries
from skimage import io
import collections as coll
import time
from scipy import ndimage as ndi
from skimage.segmentation import slic
from skimage.segmentation import slicP
from PIL import Image
from skimage.util import img_as_float
import numpy as np
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from skimage.segmentation._parallel_slic import _slic_cython
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
import os
import sys
import boto3


#spark context
from pyspark import SparkConf
from pyspark import SparkContext
#sc = SparkContext(master='local['+core+']',appName='test')

conf = SparkConf()
conf.setMaster('yarn-client')
conf.setAppName('spark-yarn')
sc=SparkContext(conf=conf)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(sc.master)

#path = "/home/ubuntu/superpixel-benchmark/data/BSDS500/images/all"
path = "s3://aws-emr-resources-846035848117-us-west-2/all"
'''
file_list = os.listdir(path)
file_list_img = [file for file in file_list if file.endswith(".jpg")]
'''


client = boto3.client('s3')
paginator = client.get_paginator('list_objects_v2')

response_iterator = paginator.paginate(
    Bucket='aws-emr-resources-846035848117-us-west-2',
    Prefix='all'
)
file_list=[]
for page in response_iterator:
    for content in page['Contents']:
        file_list.append(content['Key'][4:])
file_list_img = [file for file in file_list if file.endswith(".jpg")]


#split size
# 폴더는 overlap+1
#number of partition
time_dict = {}
pre_dict = {}
overlap = 0

s3 = boto3.resource('s3',region_name='us-west-2')
bucket = s3.Bucket('aws-emr-resources-846035848117-us-west-2')


for OriginSegments in [500]:
    testnum = 0
    for img_name in file_list_img:

        name = img_name[:-4]
        #원본 이미지 메타데이터(결과 출력용)

        obj = bucket.Object('all/'+img_name)
        file_stream = io.BytesIO()
        obj.download_fileobj(file_stream)

        images = Image.open(file_stream)

        #원본 이미지 크기
        (img_w,img_h) = images.size
        depth = 1
        #mapping 하는데 필요한 lsit 만들기
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        index = 0
        listi = []
        
        img_shape = "h"
        if img_h<img_w:
            img_shape = "w"
        for partition in [int(core)]:

            if str(OriginSegments)+img_shape+':'+str(partition) not in pre_dict:
                #grid 분할하는데 필요한 정보
                image = img_as_float(images)
                segments = slic(image,n_segments=OriginSegments,sigma=5,max_iter=1)
                init_list = []
                dist_list = []
                for i,v in enumerate(segments):
                    if v[0] not in init_list:
                        init_list.append(v[0])
                        dist_list.append(i)
                #ilist 생성
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
                pre_dict[str(OriginSegments)+img_shape+':'+str(partition)] = split_list
                pre_dict[str(OriginSegments)+img_shape+':'+str(partition)+"h"] = height
                pre_dict[str(OriginSegments)+img_shape+':'+str(partition)+"w"] = len(np.unique(segments[0]))
                
                
                
            sp_list = pre_dict[str(OriginSegments)+img_shape+':'+str(partition)]
            seg_h = pre_dict[str(OriginSegments)+img_shape+':'+str(partition)+"h"]
            seg_w = pre_dict[str(OriginSegments)+img_shape+':'+str(partition)+"w"]
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


            if img_shape == 'h':
                if OriginSegments == 500:
                    pi_list = [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 13]
                elif OriginSegments == 1000:
                    pi_list = [13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

                elif OriginSegments == 1500:
                    pi_list = [11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
                elif OriginSegments == 2000:
                    pi_list = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 13]
            else:
                if OriginSegments == 500:
                    pi_list = [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 15]
                elif OriginSegments == 1000:
                    pi_list = [13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 8]
                elif OriginSegments == 1500:
                    pi_list = [11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
                elif OriginSegments == 2000:
                    pi_list = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6]
            #ilist 생성
            imsi = 0
            split_list = []
            for i in range(partition):
                split_temp = 0
                for j in range(imsi,imsi+sp_list[i]):
                    split_temp += pi_list[j]
                split_list.append(split_temp)
                imsi += sp_list[i]
                
            listi.append(index)
            index += split_list[i]

            
            
            
            
            #mapping 하는데 필요한 k 만들기
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            #file list
            k=[]
            for i in range(partition):
                if i==0 or i == partition-1:
                    k.append([name+'_'+str(i),sp_list[i]*seg_w+min(overlap,sp_list[i])*seg_w])
                else:
                    k.append([name+'_'+str(i),sp_list[i]*seg_w+min(overlap,sp_list[i])*seg_w*2])


            #slavenode 작업 전달
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            del images
            file_stream.__exit__()
            #이미지를 float으로 변환
            #images = img_as_float(images)
            #map 함수
            def Map(k):
                #time check
                start_time = time.time()

                #img = Image.open('/home/ubuntu/superpixel-benchmark/data/BSDS500/disjoint/'+str(OriginSegments)+'/'+str(partition)+'/'+ k[0] +'.jpg')
                
                s3 = boto3.resource('s3',region_name='us-west-2')
                bucket = s3.Bucket('aws-emr-resources-846035848117-us-west-2')
                obj = bucket.Object('disjoint/'+img_shape+"/"+str(OriginSegments)+'/'+str(partition)+'/'+ k[0] +'.jpg')
                file_stream = io.BytesIO()
                obj.download_fileobj(file_stream)

                image = Image.open(file_stream)
                image = img_as_float(img)

                segments,distances = slicP(image, n_segments = k[1], sigma = 5)

                finish_time = time.time()

                file_stream.__exit__()
                del image
                
                return segments, distances, start_time, finish_time, k[0][-1]
            #times, segments, distances 를 리턴받는다.
            stime = time.time()
            datas = sc.parallelize(k,partition).map(Map).collect()

            ftime = time.time()

            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            
            
            #원본 이미지 메타데이터로 dimension 전송
            dimension = [depth,img_h,img_w]
            dimension = np.ascontiguousarray(dimension)

            #segments, distances 리스트 만들기
            indlist = [[],[]]
            tempindex = 0
            for i in datas:

                indlist[0] += [int(i[4])]
                indlist[1].append(i[0][np.newaxis,:][0].shape[0])
                if tempindex is 0:
                    seg_array = i[0][np.newaxis,:][0]
                    dist_array = i[1][0]
                else:
                    seg_array = np.r_[seg_array,i[0][np.newaxis,:][0]]
                    dist_array = np.r_[dist_array,i[1][0]]

                tempindex +=1

            # array broadcast 불가 -> 합쳐서 전송하고 다시 분리    
            # 분리할때 인덱스대로 분리하고 겉에 []를 씌워줘서 3차원 형태로 만들어야함
            #바뀐점은 한줄로 펴진 array형태로 전달하고, indlist에서 한차원 늘어남.

            #slic_master - 취합 코드
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


            
            #extension = int(split_list[i]*percent/100)
            sp_acc=[]
            acc_temp = 0
            for v in sp_list:
                sp_acc.append(acc_temp)
                acc_temp += v
            result = np.ascontiguousarray(seg_array[np.newaxis,:])
            fftime = time.time()
            
            
            #label 결속 처리
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if 1:
                segment_size = depth * img_h * img_w / OriginSegments
                min_size = int(0.5 * segment_size)
                max_size = int(3 * segment_size)
                labels = _enforce_label_connectivity_cython(result,min_size,max_size)

            #slic_master - 시간체크
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


            #time check
            import pandas as pd
            data = pd.DataFrame(columns=['node','time'])
            number = 0
            node = 1
            datat = pd.DataFrame(columns=['node','time'])
            pd.options.display.float_format = '{:.6f}'.format
            for i in datas:
                datat.loc[number]=['node '+str(node),i[3]-i[2]]

                data.loc[number]=['node '+str(node)+' start',i[2]]
                number += 1
                data.loc[number]=['node '+str(node)+' finish',i[3]]
                number += 1
                node+=1
            data = data.sort_values(by=['time'], axis=0)

            # 시간ㅊ[크]
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if img_shape+str(OriginSegments)+str(partition)+'whole' not in time_dict.keys():
                time_dict[img_shape+str(OriginSegments)+str(partition)+'whole'] = fftime-stime
                time_dict[img_shape+str(OriginSegments)+str(partition)+'master'] = fftime-ftime
            else:
                time_dict[img_shape+str(OriginSegments)+str(partition)+'whole'] += fftime-stime
                time_dict[img_shape+str(OriginSegments)+str(partition)+'master'] += fftime-ftime
            del datas
            del labels
            del result
                
                
for k,v in time_dict.items():
    if k[0] == 'h':
        print(k)
        print(v)
for k,v in time_dict.items():
    if k[0] == 'w':
        print(k)
        print(v)



