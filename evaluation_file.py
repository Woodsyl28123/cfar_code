import xml.etree.ElementTree as ET
import matplotlib.font_manager as fm

from cfar_detection_file import *

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

# def evaluate_cfar(img_result, target_boxes, xml_file):
#
#     '''
#             :param
#                 img_result: 输入有标注框的预测图
#                 target_boxes: img_result的标注框位置坐标
#                 xml_file: 预测图对应的原图的xml文件
#             :return:
#                 target_true: 真实目标原数目
#                 tp: 正确检测到的正样本数
#                 fp: 虚警
#                 fn: 漏警
#     '''
#
#     # CFAR算法
#     # result = cfar_global(img)
#
#     # 解析xml文件，获取真实的目标位置信息并存储于true_boxes
#     tree = ET.parse(xml_file) # 解析xml文件
#     root = tree.getroot() # 获取根节点
#     true_boxes = [] # 存储真实目标的位置信息
#     target_true = 0 # 真实目标原数目
#     for obj in root.iter('object'): # 遍历xml文件中的目标节点
#         name = obj.find('name').text # 目标名称
#         if name != 'ship':  # 只考虑船只目标
#             continue
#         target_true += 1
#         box = obj.find('bndbox') # 目标位置信息
#         x1 = int(box.find('xmin').text) # 左上角x坐标
#         y1 = int(box.find('ymin').text) # 左上角y坐标
#         x2 = int(box.find('xmax').text) # 右下角x坐标
#         y2 = int(box.find('ymax').text) # 右下角y坐标
#         true_boxes.append((x1, y1, x2, y2)) # 存储目标位置信息
#
#     # 计算虚警、漏警和检测准确率
#     tp = 0 # true positive，正确检测到的正样本数
#     fp = 0 # 虚警
#     fn = 0 # 漏警
#
#     # 遍历预测目标的位置信息
#     for pred_box in target_boxes:
#         iou_max = 0 # 初始化最大的IOU为0
#         for true_box in true_boxes: # 遍历真实目标的位置信息
#             iou = calc_iou(pred_box, true_box) # 计算预测目标和真实目标的IOU
#             if iou > iou_max: # 更新最大的IOU
#                 iou_max = iou
#         if iou_max >= 0.5:  # TP，如果最大IOU大于等于0.5，则认为是正确检测到的正样本，即准确数
#             tp += 1
#         else:  # FP，如果最大IOU小于0.5，则认为是错误检测到的正样本，即虚警
#             fp += 1
#
#     # 遍历真实目标的位置信息
#     for true_box in true_boxes:
#         iou_max = 0 # 初始化最大的IOU为0
#         for pred_box in target_boxes: # 遍历预测目标的位置信息
#             iou = calc_iou(pred_box, true_box) # 计算预测目标和真实目标的IOU
#             if iou > iou_max: # 更新最大的IOU
#                 iou_max = iou
#         if iou_max < 0.5:  # FN，如果最大IOU小于0.5，则认为是未检测到的正样本
#             fn += 1
#
#
#     # p = tp / (tp + fp) # 精确率（precision）
#     # r = tp / (tp + fn) # recall
#     # f1 = 2 * p * r / (p + r)
#     return target_true, tp, fp, fn

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
        if iou_max >= 0.4:  # TP，如果最大IOU大于等于0.4，则认为是正确检测到的正样本，即准确数
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

        if iou_max < 0.1:  # FN，如果最大IOU小于0.5，则认为是未检测到的正样本
            fn += 1
            x1, y1, x2, y2 = true_box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 计算矩形框中心点坐标
            cv2.circle(img_result, (cx, cy), 8, (0, 0, 255), 2)  # 画出圆形区域
            # cv2.circle(img_result, (cx, cy), 5, (0, 0, 255), -1)

    img_result_revised = img_result

    return img_result_revised, target_true, tp, fp, fn

if __name__ == '__main__':

    save_dir = r"G:\SHIP_Project\CFAR_result\result_cfar_global"  # 存储结果的路径
    img_dir = r"G:\SSDD_Dataset\ssdd_inandoff\JPEGImages_offshore"  # 存储图像数据的路径

    if not os.path.exists(save_dir):  # 如果存储结果的路径不存在，则创建该路径
        os.makedirs(save_dir)

    images = glob.glob(img_dir + '/*.jpg')  # 获取所有图像数据的路径
    num_images = len(images)
    print(num_images)
    time_sum = 0
    acc = []
    FOM = []
    for filepath in images:  # 遍历所有图像数据
        [_, tmp_img_name] = os.path.split(filepath)  # 分离文件路径和文件名
        [tmp_img_name_raw, _] = os.path.splitext(tmp_img_name)  # 分离文件名和文件扩展名，获取原始文件名
        save_path = os.path.join(save_dir, tmp_img_name_raw + '_ship_global.jpg')  # 拼接存储结果的文件路径

        filename, ext = os.path.splitext(os.path.basename(filepath))
        xml_file = r'G:\SSDD_Dataset\ssdd_inandoff\Annotations\\' + filename + '.xml'

        clutter_win = 55
        protect_win = 40
        Pfa = 1e-6
        minShip = 50
        # distribution = 'gaussian'
        distribution = 'rayleigh'
        time, img_bin = cfar_global(filepath, save_path)
        # time, img_bin = cfar_shixin(filepath, save_path, win_size=50, step_size=8)
        # time, img_bin = detection_cfar(filepath, save_path, clutter_win, protect_win, distribution, minShip, Pfa)
        time_sum += time  # 处理图像的时间叠加

        img_result, target_boxes = outline_extract(img_bin, k=3)
        img_result_revised, target_true, tp, fp, fn = evaluate_cfar(xml_file, img_bin, k=3)
        acc.append(tp / target_true)
        FOM.append(tp / (target_true + fp))

        print('实际存在的目标数 : {}'.format(target_true))
        print('正确检测数目 : {}，虚警数目 : {}，漏警数目 : {}'.format(tp, fp, fn))
        print('准确率 : {}'.format(tp / target_true))
        print('品质因数 : {}'.format(tp / (target_true + fp)))

    m_acc = np.mean(acc)
    m_FOM = np.mean(FOM)
    print('平均准确率m_acc : {}'.format(m_acc))
    print('平均品质因数m_FOM : {}'.format(m_FOM))
    print('Total time is : {}'.format(time_sum))
