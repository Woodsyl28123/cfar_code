import xml.etree.ElementTree as ET
import matplotlib.font_manager as fm

from cfar_detection import *

def xml_analysis(xml_file):

    '''
            :param
                xml_file: 输入为xml文件
            :return:
                true_boxes: 真实目标的位置信息，每个目标及其四个坐标
                target_true: 真实目标数目
    '''

    tree = ET.parse(xml_file)  # 解析xml文件
    root = tree.getroot()  # 获取根节点
    true_boxes = []  # 存储真实目标的位置信息
    target_true = 0  # 真实目标原数目
    for obj in root.iter('object'):  # 遍历xml文件中的目标节点
        name = obj.find('name').text  # 目标名称
        if name != 'ship':  # 只考虑船只目标
            continue
        target_true += 1
        box = obj.find('bndbox')  # 目标位置信息
        x1 = int(box.find('xmin').text)  # 左上角x坐标
        y1 = int(box.find('ymin').text)  # 左上角y坐标
        x2 = int(box.find('xmax').text)  # 右下角x坐标
        y2 = int(box.find('ymax').text)  # 右下角y坐标
        true_boxes.append((x1, y1, x2, y2))  # 存储目标位置信息

    return true_boxes, target_true

def outline_extract(img_bin,k):

    '''
        :param
            img_bin: 输入的二值图，单张图像
            k: kernel大小
        :return:
            img_result: 在img_bin上标注最小外接矩形的图像
            target_boxes: 所有最小外接矩形的4个顶点坐标集合
    '''

    # 1.导入图片
    img_bin = img_bin.astype(np.uint8)

    # 2.消除干扰像素点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))  # 定义结构元素
    binary = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)  # 开运算
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭运算
    img_bin = binary

    # 3.连通域分析
    contours, hierarchy = cv2.findContours(img_bin,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # 4.获取每个连通区域的最小外接矩形并打印
    img_result = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    target_boxes = []
    for cnt in contours:
        # 使用轮廓的面积进行筛选，只保留面积大于50的轮廓
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        min_rect = cv2.minAreaRect(cnt) # 返回值min_rect为一个元组，包含最小外接矩形的中心点坐标、宽度、高度和旋转角度信息
        # print("返回值min_rect:\n", min_rect)
        rect_points = cv2.boxPoints(min_rect)
        # print("返回值rect_points:\n", rect_points)
        rect_points = np.int0(rect_points) # rect_points则是一个numpy数组，包含最小外接矩形的4个顶点坐标
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2        # 计算矩形框中心点坐标
        target_boxes.append((x, y, x + w, y + h))

        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.drawContours(img_result, [rect_points], 0, (0, 0, 255), 1) # 绘制旋转框
        # cv2.circle(img_result, (cx, cy), 2, (0, 0, 255), 2)  # 画出矩形区域的中心点

    m = 0
    with open('target_boxes.txt', 'w') as f:
        for x1, y1, x2, y2 in target_boxes:
            m += 1
            f.write(f'position_{m}: {x1},{y1},{x2},{y2}\n')
        f.write(f'sum:{len(target_boxes)} ship')

    return img_result, target_boxes

def calc_iou(box1, box2):
    # 计算两个矩形框的IOU
    x1 = max(box1[0], box2[0]) # 两个矩形框左上角x坐标的最大值
    y1 = max(box1[1], box2[1]) # 两个矩形框左上角y坐标的最大值
    x2 = min(box1[2], box2[2]) # 两个矩形框右下角x坐标的最小值
    y2 = min(box1[3], box2[3]) # 两个矩形框右下角y坐标的最小值
    inter_area = max(0, x2 - x1) * max(0, y2 - y1) # 两个矩形框的交集面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) # 第一个矩形框的面积
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) # 第二个矩形框的面积
    iou = inter_area / (area1 + area2 - inter_area) # IOU即交集面积除以并集面积减去交集面积
    return iou

def evaluate_cfar(xml_file, img_bin,k):

    '''
        :param
            xml_file:
            img_bin: 输入的二值图，单张图像
            k: kernel大小
        :return:
            img_result_revised: img_bin上标注正确（矩形）、虚警（三角形）、漏警（圆形）
            target_true: 目标数
            tp: 正确检测到的正样本数
            fp: 虚警
            fn: 漏警

    '''


    # 1.导入图片
    img_bin = img_bin.astype(np.uint8)

    # 2.消除干扰像素点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))  # 定义结构元素
    binary = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)  # 开运算
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭运算
    img_bin = binary

    # # 形态学梯度
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)) # 定义结构元素
    # gradient = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel) # 计算形态学梯度
    # edge_mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY)[1] # 二值化形态学梯度，生成边缘掩模
    # filled_img = cv2.inpaint(img_bin, edge_mask, k, cv2.INPAINT_TELEA)# 对原始图像进行掩模操作，填充细小空洞
    # img_bin = filled_img

    # 3.连通域分析
    contours, hierarchy = cv2.findContours(img_bin,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # 4.获取每个连通区域的最小外接矩形并打印
    img_result = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    true_boxes, target_true= xml_analysis(xml_file)  # 真实目标的位置信息，每个目标及其四个坐标

    target_boxes = [] # 预测目标的位置信息，预测的每个目标及其左上角和右下角坐标
    tp = 0
    fp = 0
    fn = 0
    for cnt in contours:
        # 使用轮廓的面积进行筛选，只保留面积大于50的轮廓
        area = cv2.contourArea(cnt)
        # if area < 20:
        #     continue

        min_rect = cv2.minAreaRect(cnt) # 返回值min_rect为一个元组，包含最小外接矩形的中心点坐标、宽度、高度和旋转角度信息
        # print("返回值min_rect:\n", min_rect)
        rect_points = cv2.boxPoints(min_rect)
        # print("返回值rect_points:\n", rect_points)
        rect_points = np.int0(rect_points) # rect_points则是一个numpy数组，包含最小外接矩形的4个顶点坐标
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2        # 计算矩形框中心点坐标
        pred_box = (x, y, x + w, y + h)
        target_boxes.append(pred_box)

        iou_max = 0  # 初始化最大的IOU为0
        for true_box in true_boxes:  # 遍历真实目标的位置信息
            iou = calc_iou(pred_box, true_box)  # 计算预测目标和真实目标的IOU
            if iou > iou_max:  # 更新最大的IOU
                iou_max = iou
        if iou_max >= 0.5:  # TP，如果最大IOU大于等于0.5，则认为是正确检测到的正样本，即准确数
            tp += 1
            cv2.rectangle(img_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:  # FP，如果最大IOU小于0.5，则认为是错误检测到的正样本，即虚警
            fp += 1
            triangle_points = np.array([[x + w / 2, y - h /2], [x, y + h], [x + w, y + h]])
            triangle_points = np.int0(triangle_points)
            cv2.drawContours(img_result, [triangle_points], 0, (0, 255, 0), 2)
            # cv2.drawContours(img_result, [rect_points], 0, (0, 255, 0), 2)

    # 遍历真实目标的位置信息
    for true_box in true_boxes:
        iou_max = 0  # 初始化最大的IOU为0
        for pred_box in target_boxes:  # 遍历预测目标的位置信息
            iou = calc_iou(pred_box, true_box)  # 计算预测目标和真实目标的IOU
            if iou > iou_max:  # 更新最大的IOU
                iou_max = iou

        if iou_max < 0.5:  # FN，如果最大IOU小于0.5，则认为是未检测到的正样本，即漏警
            fn += 1
            x1, y1, x2, y2 = true_box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 计算矩形框中心点坐标
            cv2.circle(img_result, (cx, cy), 8, (0, 0, 255), 2)  # 画出圆形区域
            # cv2.circle(img_result, (cx, cy), 5, (0, 0, 255), -1)

    img_result_revised = img_result

    return img_result_revised, target_true, tp, fp, fn

if __name__ == '__main__':

    # 读取图像和目标框标注
    # img = cv2.imread(r"G:\SSDD_Dataset\ssdd_inandoff\JPEGImages\001159.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(r"G:\SHIP_Project\SEA_LAND_code\result_Otsu&&K-means\result_img\exp_1159\final.jpg", cv2.IMREAD_GRAYSCALE)
    xml_file = r"G:\SSDD_Dataset\ssdd_inandoff\Annotations\001159.xml"
    img_box = cv2.imread(r"G:\SSDD_Dataset\ssdd_inandoff\Result_imgs\001159.jpg")

    # CFAR检测和目标框提取
    clutter_win = 45
    protect_win = 30
    Pfa = 1e-6
    minShip = 10
    # 'rayleigh' or 'gaussian' or 'gamma' or 'lognormal'
    distribution = 'lognormal'
    img_bin = cfar_global(img)
    # img_bin = cfar_shixin(img)
    # img_bin = detection_cfar(img, clutter_win, protect_win, distribution, minShip, Pfa)
    # img_result, target_boxes = outline_extract(img_bin, k=1)
    img_result_revised, target_true, tp, fp, fn = evaluate_cfar(xml_file, img_bin, k=2)
    FOM = tp / (target_true + fp)
    print('实际存在的目标数 : {}'.format(target_true))
    print('正确检测数目 : {}，虚警数目 : {}，漏警数目 : {}'.format(tp, fp, fn))
    print('品质因数 : {}'.format(FOM))

    # 绘制子图
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Sea - land - split Image')
    # Sea - land - split Original
    axs[0, 0].axis('off')  # 关闭坐标轴
    axs[0, 1].imshow(img_box)
    axs[0, 1].set_title('Ground Truth')
    axs[0, 1].axis('off')  # 关闭坐标轴
    axs[1, 0].imshow(img_bin, cmap='gray')
    axs[1, 0].set_title('CFAR Detection')
    axs[1, 0].axis('off')  # 关闭坐标轴
    axs[1, 1].imshow(img_result_revised)
    axs[1, 1].set_title('CFAR_boxed')
    axs[1, 1].axis('off')  # 关闭坐标轴


    # 指定中文字体
    font_path = r"c:\windows\fonts\SIMLI.TTF"
    font_prop = fm.FontProperties(fname=font_path, size= 15)

    # 显示评估结果
    plt.figtext(0.5, 0.1, '实际存在的目标数: {} \n'
                           '正确检测数目: {} 虚警数目: {} 漏警数目: {}\n'
                           '品质因数FOM: {:.3f}'.format(target_true, tp, fp, fn, FOM),
                ha='center', fontsize=14, fontproperties=font_prop)


    # 调整子图之间的距离和布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

    # img = cv2.imread(r'./data/test/offshore/001108.jpg', cv2.IMREAD_GRAYSCALE)
    # xml_file = r'./data/BBox_SSDD/voc_style/Annotations/001108.xml'
    # img_box = cv2.imread(r'./data/BBox_SSDD/voc_style/Result_imgs/001108.jpg')
    #
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.figure()
    # plt.imshow(img_box)
    #
    # clutter_win = 45
    # protect_win = 30
    # Pfa = 1e-6
    # minShip = 50
    # distribution = 'gaussian'
    # # img_bin = cfar_global(img)
    # # img_bin = cfar_shixin(img)
    # img_bin = detection_cfar(img, clutter_win, protect_win, distribution, Pfa=1e-6, minShip=50)
    #
    # img_result, target_boxes = outline_extract(img_bin, k=3)
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    #
    # # target_true, tp, fp, np = evaluate_cfar(img_result, target_boxes, xml_file)
    # img_result_revised, target_true, tp, fp, fn = evaluate_cfar(xml_file, img_bin, k=3)
    # FOM = tp / (target_true + fp)
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img_result_revised, cv2.COLOR_BGR2RGB))
    # print('实际存在的目标数 : {}'.format(target_true))
    # print('正确检测数目 : {}，虚警数目 : {}，漏警数目 : {}'.format(tp, fp, fn))
    # print('品质因数 : {}'.format(FOM))
    #
    # plt.show()