import os # 导入操作系统模块
import cv2 # 导入OpenCV模块
import matplotlib.pyplot as plt
import numpy as np # 导入numpy模块

from scipy.special import gdtrix # 导入scipy库的special模块中的gdtrix函数
import random # 导入random模块
import time # 导入time模块
import glob # 导入glob模块
from skimage.measure import label # 从skimage库中导入measure模块中的label函数
from skimage import morphology # 导入skimage库中的morphology模块
from PIL import Image
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import kstwobign
from scipy.stats import gamma
# from scipy.special import gamma
from osgeo import gdal # 导入gdal模块
from osgeo import osr # 导入osr模块


# 全局CFAR
def cfar_global(img, save_path):

    '''
    :param
        img: 输入图像，单张SAR图像
    :return:
        result * 255, 即二值图
    '''
    globalPfa = 1e-6 # 全局虚警概率设为10的负六次方
    minShip = 50 # 最小的船只面积，去除像素小于50的地方

    start = time.time() # 记录开始时间
    data_pre = cv2.imread(img)  # 打开指定的地理数据集
    gray = cv2.cvtColor(data_pre, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度格式
    data = np.array(gray)  # 将灰度图像转化为numpy数组
    # data = np.array(img)# 将灰度图像转化为numpy数组
    a = data[data > 0] # 选取数据中大于0的部分
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
    thresholds_global = gdtrix(p2, p1, 1 - globalPfa)  # 计算全局门限值
    end3 = time.time()  # 记录结束时间
    spend3 = end3 - start3  # 计算时间差
    print('thresholds_global : {}'.format(spend2))  # 输出运行时间

    result = np.int8(data > thresholds_global) # 对图像进行二值化

    L = label(result , connectivity=2) # 对二值化后的图像进行连通区域分析
    result  = morphology.remove_small_objects(L,minShip) # 移除小物体
    result[result!=0] = 1 # 最终二值化结果
    cv2.imwrite(save_path, result * 255)
    # cv2.imwrite('target_mask.png',result*255) # 将结果保存为图像文件
    end = time.time() # 记录结束时间
    print(' all time: {}'.format(end-start))

    return end-start, result * 255

def cfar_shixin(filepath, save_path, win_size, step_size):

    img1 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    N, M = img1.shape  # 图像尺寸

    data_pre = cv2.imread(filepath)  # 打开指定的地理数据集
    gray = cv2.cvtColor(data_pre, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度格式
    img = np.array(gray)  # 将灰度图像转化为numpy数组

    # win_size = 50 # 滑动窗口大小
    # step_size = 8 # 步长
    Pfa = 1e-6 # 恒虚警概率
    minShip = 50

    detections = np.zeros((N, M))
    start = time.time()  # 记录开始时间
    # 对于每个窗口进行 CFAR 检测
    for i in range(0, N - win_size, step_size):
        for j in range(0, M - win_size, step_size):
            # 获取当前窗口
            window = img[i:i+win_size, j:j+win_size]

            data = np.array(window)
            a = window[window>0]
            sea = a[a < 2 *np.mean(a)]

            '''随机选取数据做拟合'''
            start1 = time.time()  # 记录时间
            sea = list(sea)  # 将数据转化为列表
            t = min(10000, len(sea))  # 选取最小值为10000和列表长度之间的较小值
            x = random.sample(sea, t)  # 从列表中随机选取t个样本
            end1 = time.time()  # 记录时间
            spend1 = end1 - start1  # 计算时间差
            # print('random.sample time : {}'.format(spend1))  # 输出随机抽样所用的时间

            start2 = time.time()  # 记录时间
            mu = np.mean(x)  # 计算样本的平均值
            var = np.var(x)  # 计算样本的方差
            p1 = mu ** 2 / var  # 计算p1，是一个权重系数，表示数据中的信号强度
            p2 = mu / var  # 计算p2，表示数据中的噪声强度
            end2 = time.time()  # 结束计时
            spend2 = end2 - start2  # 计算时间差
            # print('mean-var time : {}'.format(spend2))

            '''该算法采用全局阈值'''
            start3 = time.time()  # 记录起始时间
            thresholds_global = gdtrix(p2, p1, 1 - Pfa)  # 计算全局门限值
            end3 = time.time()  # 记录结束时间
            spend3 = end3 - start3  # 计算时间差
            # print('thresholds_global : {}'.format(spend2))  # 输出运行时间

            result = np.int8(data > thresholds_global)  # 对图像进行二值化

            # 对窗口内的像素进行二值化并记录检测结果
            result = np.int8(data > thresholds_global)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

            L = label(result, connectivity=2) # 对二值化后的图像进行连通区域分析
            L = np.array(L, dtype=bool)
            result = morphology.remove_small_objects(L, minShip) # 移除小物体
            detections[i:i + win_size, j:j + win_size] += result


    end = time.time()  # 记录结束时间
    print('all time: {}'.format(end - start))
    detections[detections != 0] = 1  # 最终二值化结果
    cv2.imwrite(save_path, detections * 255)

    return end - start, detections * 255

# 获取背景窗口的杂波样本数据
def detection_get_clutter(sar_data, clutter_win, protect_win, x, y):
    """
    获取背景窗口的杂波样本数据

    输入参数：
        sar_data：二维的 SAR 数据
        clutter_win：背景窗口半长
        protect_win：保护窗口半长
        x：待检测点的行坐标（横坐标）
        y：待检测点的列坐标（纵坐标）

    输出参数：
        data_out：一维杂波样本数据
    """
    # %
    # % | ———————————————————————— |
    # % | ——————————-1———————————— |
    # % |   |                  |   |
    # % |   |     |——————|     |   |
    # % | 3 |     |target|     | 4 |
    # % |   |     |——————|     |   |
    # % |   |                  |   |
    # % |   |                  |   |
    # % | ———————————————————————— |
    # % | ——————————-2———————————— |
    # %
    data_part1 = sar_data[x - clutter_win:x - protect_win - 1, y - clutter_win:y + clutter_win].flatten()  # 上
    data_part2 = sar_data[x + protect_win + 1:x + clutter_win, y - clutter_win:y + clutter_win].flatten()  # 下
    data_part3 = sar_data[x - protect_win:x + protect_win, y - clutter_win:y - protect_win - 1].flatten()  # 左
    data_part4 = sar_data[x - protect_win:x + protect_win, y + protect_win + 1:y + clutter_win].flatten()  # 右

    data_out = np.concatenate([data_part1, data_part2, data_part3, data_part4])  # 结果是一维数组

    return data_out

# 基于不同杂波模型分布的检测阈值计算
def detection_count_th(clutter_data, Pfa, distribution):
    """
        基于不同分布的检测阈值计算

        输入参数：
            clutter_data：杂波样本数据
            Pfa：虚警概率
            distribution：分布类型，包括'gamma'（默认）、'rayleigh'和'gaussian'

        输出参数：
            threshold：检测阈值
        """

    if distribution == 'rayleigh':
        sigma = np.sqrt(np.mean(clutter_data ** 2) / 2)  # 估计分布参数，瑞利分布的标准差
        threshold = sigma * np.sqrt(-2 * np.log(Pfa))  # 计算阈值

    elif distribution == 'gaussian':
        mean_val = np.mean(clutter_data)  # 估计分布参数，均值
        std_val = np.std(clutter_data)  # 方差
        times = norm.ppf(1 - Pfa)
        threshold = mean_val + times * std_val  # 计算阈值

    elif distribution == 'lognormal':
        mean_val = np.mean(clutter_data)  # 估计分布参数，均值
        std_val = np.std(clutter_data)  # 方差

        # 计算对数正态分布的参数
        sigma = np.sqrt(np.log(std_val ** 2 / mean_val ** 2 + 1))
        mu = np.log(mean_val) - sigma ** 2 / 2

        # 计算阈值
        log_threshold = mu + lognorm.ppf(1 - Pfa, sigma)
        if log_threshold > np.log(np.finfo(float).max):
            threshold = np.finfo(float).max
        else:
            threshold = np.exp(log_threshold)

    elif distribution == 'gamma':
        mean_val = np.mean(clutter_data)  # 估计分布参数，均值
        std_val = np.std(clutter_data)  # 方差
        num_train_cells = len(clutter_data)
        a = (mean_val / std_val) ** 2  # 计算伽马分布的形状参数
        scale = std_val ** 2 / mean_val  # 计算伽马分布的尺度参数
        threshold = gamma.ppf(1 - Pfa, a=a, scale=scale)  # 计算阈值

    else:
        raise ValueError("distribution should be one of ['gamma', 'rayleigh', 'gaussian']")

    return threshold

# 双参数CFAR检测
def detection_cfar(filepath, save_path, clutter_win, protect_win, distribution, minShip, Pfa):
    """
    基于高斯分布的 CFAR 检测算法（双参数 CFAR 检测）

    输入参数：
        filepath：二维的 SAR 数据
        clutter_win：背景窗口半长
        protect_win：保护窗口半长
        Pfa: 虚警率
        minShip: 需要取出的船只最小面积像素

    输出参数：
        time
        data_out：检测后的二值图像
    """
    start = time.time()
    # print('CFAR检测开始')
    data_pre = cv2.imread(filepath)  # 打开指定的地理数据集
    gray = cv2.cvtColor(data_pre, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度格式
    img = np.array(gray)  # 将灰度图像转化为numpy数组
    sar_data = img.astype(np.float64)

    start1 = time.time()  # 记录时间
    '''
    对 sar_data_backup 进行边缘补充
    即将 SAR 数据填充到新矩阵中,填充的方式是：先在上、下、左、右四个边缘分别填充背景窗口大小的数据，然后将 SAR 数据填充到中心区域。
    '''

    [r, c] = sar_data.shape
    # sar_data_backup = np.copy(sar_data)
    sar_data_backup = np.zeros((r + 2 * clutter_win, c + 2 * clutter_win))  # SAR 数据边缘补充
    sar_data_backup[0:clutter_win, clutter_win:c + clutter_win] = np.tile(sar_data[0, :], (clutter_win, 1))
    sar_data_backup[r + clutter_win:, clutter_win:c + clutter_win] = np.tile(sar_data[r - 1, :], (clutter_win, 1))
    sar_data_backup[clutter_win:r + clutter_win, clutter_win:c + clutter_win] = sar_data[:,:]
    sar_data_backup[:, 0:clutter_win] = np.tile(sar_data_backup[:, clutter_win:clutter_win + 1], (1, clutter_win))
    sar_data_backup[:, clutter_win + c:] = np.tile(sar_data_backup[:, clutter_win + c - 1:clutter_win + c], (1, clutter_win))

    end1 = time.time()  # 记录时间
    spend1 = end1 - start1  # 计算时间差
    print('padding time : {}'.format(spend1))  # 输出随机抽样所用的时间

    row, col = sar_data_backup.shape
    data_out = np.zeros((row, col))  # 初始化检测结果矩阵

    start2 = time.time()  # 记录时间
    # 滑窗移动
    for x in range(clutter_win, row - clutter_win):
        for y in range(clutter_win, col - clutter_win):
            clutter_data = detection_get_clutter(sar_data_backup, clutter_win, protect_win, x, y)  # 获取杂波
            threshold = detection_count_th(clutter_data, Pfa, distribution)
            if sar_data_backup[x, y] > threshold:
                data_out[x, y] = 1  # 检测结果（二值）
            else:
                data_out[x, y] = 0  # 检测结果（二值）

    end2 = time.time()  # 结束计时
    spend2 = end2 - start2  # 计算时间差
    print('window.sliding time : {}'.format(spend2))

    start3 = time.time()  # 记录起始时间
    data_out = data_out[clutter_win:r+clutter_win, clutter_win:c+clutter_win]
    L = label(data_out, connectivity=2)  # 对二值化后的图像进行连通区域分析
    L = np.array(L, dtype=bool)
    result = morphology.remove_small_objects(L, minShip)  # 移除小物体
    result[result!=0] = 1 # 最终二值化结果
    cv2.imwrite(save_path, result * 255)
    end = time.time()  # 记录结束时间
    spend = end - start
    print('all time : {}'.format(spend))
    # print('CFAR检测完成')
    # print(filename + '\n all time: {}'.format(end - start))
    return spend, result * 255

