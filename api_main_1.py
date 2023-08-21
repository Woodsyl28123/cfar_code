
# coding: utf-8

'''全局CFAR'''
# In[21]:
import os # 导入操作系统模块
import cv2 # 导入OpenCV模块
import numpy as np # 导入numpy模块
# from osgeo import osr # 导入osr模块，该模块是gdal库的一部分，主要用于处理空间参考
# from osgeo import gdal, gdalconst # 导入gdal和gdalconst模块，主要用于地理数据处理
#
# from osgeo import gdal # 再次导入gdal模块
# from osgeo import osr # 再次导入osr模块
import numpy as np # 再次导入numpy模块

from scipy.special import gdtrix # 导入scipy库的special模块中的gdtrix函数
import random # 导入random模块
import time # 导入time模块
import glob # 导入glob模块
from skimage.measure import label # 从skimage库中导入measure模块中的label函数
from skimage import morphology # 导入skimage库中的morphology模块
from PIL import Image

# 定义主函数
def api_main(filename, save_path):
    globalPfa = 1e-6 # 全局虚警概率设为10的负六次方
    minShip = 50 # 最小的船只面积，去除像素小于50的地方

    start = time.time() # 记录开始时间
    dataset = cv2.imread(filename)# 打开指定的地理数据集
    gray = cv2.cvtColor(dataset, cv2.COLOR_BGR2GRAY) # 将图像转化为灰度格式
    data = np.array(gray)# 将灰度图像转化为numpy数组

    # 以下代码对于filename里jpg图片则无法使用gdal库读取该格式的图像，因为gdal库不支持JPEG格式的图像
    # dataset = gdal.Open(filename)
    # data = np.array(dataset.GetRasterBand(1).ReadAsArray()) # 读取该数据集的第一个波段的数据，转化为numpy数组
    # info = dataset.GetGeoTransform() # 读取该数据集的空间参考信息
    # azimuthSpacingInitial = abs(info[1]) # 获取方位间隔
    # rangeSpacingInitial = abs(info[5]) # 获取距离间隔
    # spacingInitial = np.sqrt(azimuthSpacingInitial * rangeSpacingInitial) # 根据方位间隔和距离间隔计算初始间隔

    a = data[data>0] # 选取数据中大于0的部分
    sea = a[a < 2*np.mean(a)] # 将数据中小于2倍均值的部分标记为海面

    '''随机选取数据做拟合'''
    start1 = time.time() # 记录时间
    sea = list(sea) # 将数据转化为列表
    t = min(10000,len(sea)) # 选取最小值为10000和列表长度之间的较小值
    x = random.sample(sea,t) # 从列表中随机选取t个样本
    end1 = time.time() # 记录时间
    spend1 = end1-start1 # 计算时间差
    print('random.sample time : {}'.format(spend1)) # 输出随机抽样所用的时间

    start2 = time.time() # 记录时间
    mu = np.mean(x) # 计算样本的平均值
    var = np.var(x) # 计算样本的方差
    p1 = mu**2 / var  # 计算p1，是一个权重系数，表示数据中的信号强度
    p2 = mu / var  # 计算p2，表示数据中的噪声强度
    end2 = time.time()  # 结束计时
    spend2 = end2 - start2  # 计算时间差
    print('mean-var time : {}'.format(spend2))

    '''该算法采用全局阈值'''
    start3 = time.time()  # 记录起始时间
    thresholds_global = gdtrix(p2, p1, 1 - globalPfa)  # 计算全局门限值，统计分布模型采用了高斯分布
    end3 = time.time()  # 记录结束时间
    spend3 = end3 - start3  # 计算时间差
    print('thresholds_global : {}'.format(spend2))  # 输出运行时间

    result = np.int8(data > thresholds_global) # 对图像进行二值化

    L  = label(result , connectivity=2) # 对二值化后的图像进行连通区域分析
    result  = morphology.remove_small_objects(L,minShip) # 移除小物体
    result[result!=0] = 1 # 最终二值化结果
    cv2.imwrite(save_path,result*255) # 将结果保存为图像文件
    end = time.time() # 记录结束时间
    print(filename + '\n all time: {}'.format(end-start))

    return end-start

if __name__ == '__main__':

    # save_dir = '/emwusr2/jhc/cjy/cfar/cfar_result'  # 存储结果的路径
    # img_dir = '/emwusr/jhc/igarss2021/data/train/gf_4d6_imgs'  # 存储图像数据的路径
    save_dir = r'./cfar/cfar_result_2'  # 存储结果的路径
    img_dir = r'./data/BBox_SSDD/voc_style/JPEGImages_test'  # 存储图像数据的路径

    if not os.path.exists(save_dir): # 如果存储结果的路径不存在，则创建该路径
        os.makedirs(save_dir)
    # images=glob.glob(img_dir + '/*.tiff') # 获取所有图像数据的路径
    images = glob.glob(img_dir + '/*.jpg')  # 获取所有图像数据的路径
    time_sum = 0
    for filepath in images: # 遍历所有图像数据
        [_, tmp_img_name]=os.path.split(filepath)  # 分离文件路径和文件名
        [tmp_img_name_raw,_]=os.path.splitext(tmp_img_name)   # 分离文件名和文件扩展名，获取原始文件名
        save_path = os.path.join(save_dir,tmp_img_name_raw+'_ship_global.jpg') # 拼接存储结果的文件路径

        time_sum += api_main(filepath, save_path) # 处理图像数据并将结果存储

    print('Total time is : {}'.format(time_sum))

