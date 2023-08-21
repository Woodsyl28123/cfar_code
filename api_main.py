
# coding: utf-8

# In[21]:

import os
import cv2
import numpy as np
from osgeo import osr
from osgeo import gdal, gdalconst


from osgeo import gdal
from osgeo import osr
import numpy as np

from scipy.special import gdtrix
import random
import time
import glob
from skimage.measure import label
from skimage import morphology


def api_main(filename, save_path):
    globalPfa = 1e-6
    minShip = 50
    
    start = time.time()
    dataset = gdal.Open(filename) 
    data = np.array(dataset.GetRasterBand(1).ReadAsArray())   
    info = dataset.GetGeoTransform()
    azimuthSpacingInitial = abs(info[1])
    rangeSpacingInitial = abs(info[5])
    spacingInitial = np.sqrt(azimuthSpacingInitial * rangeSpacingInitial)



    a = data[data>0]
    sea = a[a < 2*np.mean(a)]


    '''随机选取数据做拟合'''
    start1 = time.time()
    sea = list(sea)
    t = min(10000,len(sea))
    x = random.sample(sea,t)
    end1 = time.time()
    spend1 = end1-start1
    print('random.sample time : {}'.format(spend1))

    start2 = time.time()
    mu = np.mean(x)
    var = np.var(x)
    p1 = mu**2/var
    p2 = mu/var
    end2 = time.time()
    spend2 = end2-start2
    print('mean-var time : {}'.format(spend2))
    
    start3 = time.time()
    thresholds_global = gdtrix(p2,p1,1 - globalPfa)
    end3 = time.time()
    spend3 = end3-start3
    print('thresholds_global : {}'.format(spend2))

  

    result = np.int8(data > thresholds_global) 
    L  = label(result , connectivity=2)
    result  = morphology.remove_small_objects(L,minShip)
    result[result!=0] = 1
    cv2.imwrite(save_path,result*255)
    end = time.time()
    print(filename + '\n all time: {}'.format(end-start))
    

    
if __name__ == '__main__':
    save_dir = '/emwusr2/jhc/cjy/cfar/cfar_result'
    img_dir = '/emwusr/jhc/igarss2021/data/train/gf_4d6_imgs'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images=glob.glob(img_dir + '/*.tiff')

    for filepath in images:
        [_, tmp_img_name]=os.path.split(filepath)
        [tmp_img_name_raw,_]=os.path.splitext(tmp_img_name) 
        save_path = os.path.join(save_dir,tmp_img_name_raw+'_ship_global.png')

        api_main(filepath, save_path)

