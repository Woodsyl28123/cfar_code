'''用于VOC数据集垂直框可视化'''

import cv2
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import matplotlib.image as img

def voc_visual(data_dir,        # 指定voc数据集所在目录
               img_dir,ann_dir, # 指定input和annotation文件夹的路径
               output_dir):     # 指定结果图片保存目录
    # 遍历annotation文件夹中所有xml文件
    for ann_filename in os.listdir(ann_dir):
        # 获取图片文件名
        img_filename = ann_filename.replace('.xml', '.jpg')
        # 构造图片文件路径
        img_path = os.path.join(img_dir, img_filename)
        # 读取图片
        img = cv2.imread(img_path)
        # 构造xml文件路径
        ann_path = os.path.join(ann_dir, ann_filename)
        # 解析xml文件
        tree = ET.parse(ann_path)
        root = tree.getroot()
        # 遍历xml文件中所有object
        for obj in root.findall('object'):
            # 获取bounding box坐标
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            # 在图片上绘制bounding box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 构造结果图片保存路径
        output_path = os.path.join(output_dir, img_filename)
        # 保存结果图片
        cv2.imwrite(output_path, img)

def figure_plot(data_dir,output_dir):
    # 设置子图布局，2行3列
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    # 加载图片并将其放置到相应的子图中
    img1 = img.imread(data_dir + '/JPEGImages/000083.jpg') # 远岸
    axs[0, 0].imshow(img1, extent=[0, 1, 0, 1])
    img2 = img.imread(output_dir + '/000083.jpg')
    axs[1, 0].imshow(img2, extent=[0, 1, 0, 1])
    img3 = img.imread(data_dir + '/JPEGImages/001096.jpg')
    axs[0, 1].imshow(img3, extent=[0, 1, 0, 1])
    img4 = img.imread(output_dir + '/001096.jpg')
    axs[1, 1].imshow(img4, extent=[0, 1, 0, 1])
    img5 = img.imread(data_dir + '/JPEGImages/000228.jpg') # 近岸
    axs[0, 2].imshow(img5, extent=[0, 1, 0, 1])
    img6 = img.imread(output_dir + '/000228.jpg')
    axs[1, 2].imshow(img6, extent=[0, 1, 0, 1])
    img7 = img.imread(data_dir + '/JPEGImages/000641.jpg')
    axs[0, 3].imshow(img7, extent=[0, 1, 0, 1])
    img8 = img.imread(output_dir + '/000641.jpg')
    axs[1, 3].imshow(img8, extent=[0, 1, 0, 1])

    # 隐藏坐标轴
    for ax in axs.flat:
        ax.axis('off')

    # 设置子图之间的距离和边距
    # wspace和hspace参数分别用于设置子图之间的水平和垂直间距，left、right、bottom和top参数用于设置子图与边界的距离
    plt.subplots_adjust(wspace=0.1, hspace=0.05, left=0.05, right=0.95, bottom=0.05, top=0.95)
    # 显示子图
    plt.show()

if __name__ == "__main__":
    # 指定voc数据集所在目录
    data_dir = r"G:\SSDD_Dataset\ssdd_inandoff"
    # 指定input和annotation文件夹的路径
    img_dir = os.path.join(data_dir, 'JPEGImages')
    ann_dir = os.path.join(data_dir, 'Annotations')
    # 指定结果图片保存目录
    output_dir = os.path.join(data_dir, 'Result_imgs')
    voc_visual(data_dir,img_dir,ann_dir,output_dir)

# 显示子图
    # figure_plot(data_dir,output_dir)

