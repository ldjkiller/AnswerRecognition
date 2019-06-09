import cv2
import numpy as np
import collections


# 图片处理模块，用来对原始图片进行处理


def change_size(path, ifChange):
    # 载入图片
    # 如果更改图片大小，则更改图片的尺寸至 （1200,1200）
    img = cv2.imread(path)
    if ifChange:
        return cv2.resize(img, (1200, 1200), interpolation=cv2.INTER_CUBIC)
    else:
        return img


def load_image(path, ifChange=True):
    # 获取图片，长， 宽，灰度图
    if type(path) == str:
        img = change_size(path, ifChange)
    else:
        img = path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    print(height, width, "hhhwww")
    return img, height, width, gray


def Sobel_gradient(blurred):
    # 计算x和y方向梯度
    return cv2.Canny(blurred, 40, 80)


def Thresh_and_blur(gradient):
    # 二值化处理
    _, thresh_blue_img = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
    return thresh_blue_img


def imgEnhance(img, a=2., b=0.5):
    # a>1时，图像对比度被放大，0<a<1时图像对比度被缩小
    # b>0时,亮度增强，b<0时对比度降低。
    O = img * float(a) + b
    O[O > 255] = 255
    O = np.round(O)
    O = O.astype(np.uint8)
    return O


def Gaussian_Blur(gray, x_=7, y_=7):
    # 高斯去噪
    return cv2.GaussianBlur(gray, (x_, y_), 0)


def doErode(img, iter):
    # 腐蚀操作
    return cv2.erode(img, None, iterations=iter)


def doDilate(img, iter):
    # 膨胀操作
    return cv2.dilate(img, None, iterations=iter)


def crop_image(img, minx, miny, height, width):
    # 截图
    return img[miny:miny + height, minx:minx + width]


def loc_point(height, start, end):
    # 根据相似三角形的性质进行分割线部分在边缘区域的坐标定位
    new_start = start[0] + (end[0] - start[0]) * (0 - start[1]) / (end[1] - start[1])
    new_end = start[0] + (end[0] - start[0]) * (height - start[1]) / (end[1] - start[1])
    return int(new_start), int(new_end)


def find_longest_line(thresh_blue_img, height, width):
    # 找到最长的分割线
    lines = cv2.HoughLines(thresh_blue_img, 1, np.pi / 180, int(height * 1 / 5))
    lines1 = lines[:, 0, :]  # 提取为为二维
    max_line = 0
    max_line_p = None
    for rho, theta in lines1[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        y0 = b * rho
        y1 = int(y0 + 2000 * (a))
        y2 = int(y0 - 2000 * (a))
        l = (y2 - y1) ** 2
        x0 = a * rho
        x1 = int(x0 + 2000 * (-b))
        x2 = int(x0 - 2000 * (-b))
        if l > max_line and 0.25 * width < (x1 + x2) / 2 < 0.75 * width:
            max_line = l
            max_line_p = x1, y1, x2, y2
    return max_line_p


def charge_connect_line(lines):
    # 找到纵坐标下连接的线段，区分出所有线段不相连的部分
    lines = sorted(lines, key=lambda x: x[1], reverse=False)
    print(lines)
    a = lines[0]
    _, y1, _, y2 = a
    aset = set(range(y1, y2, 1))
    out_p = []
    for line in lines[1:]:
        _, y1, _, y2 = line
        bset = set(range(y1, y2, 1))
        if aset & bset:
            aset = set(range(min([y1, min(aset)]), max([y2, max(aset)])))
        else:
            out_p.append([min(aset), max(aset)])
            aset = bset
    out_p.append([min(aset), max(aset)])
    return out_p


def find_cut_line(lis, height):
    # 找到部分分割线区域的位置
    # 并对其进行分块坐标位置的确定
    lis = list(map(lambda x: [x[2], x[3], x[0], x[1]] if x[1] > x[3] else x, lis))
    res = charge_connect_line(lis)
    # print("#########res",res)
    p0, p1 = res[0]
    cut_line = []
    if len(res) > 1:
        for i in res[1:]:
            p2 = i[0]
            cut_line.append((p1 + p2) / 2)
            p1 = i[1]
        return cut_line
    else:
        if p0 - height / 2 < 0:
            cut_line.append(p1 + 15)
        elif p0 - height / 2 > 0:
            cut_line.append(p0 - 15)
        return cut_line


def find_mid_long_line(o_img, thresh_blue_img, height, width):
    # 找到图中中间部分的线段
    lines = cv2.HoughLinesP(thresh_blue_img, 1, np.pi / 180, 60, minLineLength=int(height * 1 / 30), maxLineGap=4)
    line_map = collections.defaultdict(list)
    line_lis = []
    for x in lines:
        x1, y1, x2, y2 = x[0]
        l = (y2 - y1) ** 2
        if l > int(height * 1 / 15) ** 2 and 0.25 * width < (x1 + x2) / 2 < 0.75 * width:
            max_line_p = x1, y1, x2, y2
            cv2.line(o_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            line_lis.append(max_line_p)
            line_map[str(l)].append(max_line_p)
    return line_lis


def find_parts(o_img, thresh_blue_img, height, width):
    # 找到图中所有可区分的部分
    line_lis = find_mid_long_line(o_img, thresh_blue_img, height, width)
    if line_lis:
        cut_line = find_cut_line(line_lis, height)
        parts = []
        start_point = [0, 0]
        for l in cut_line:
            cv2.line(o_img, (0, int(l)), (width, int(l)), (0, 0, 255), 2)
            parts.append([start_point[0], start_point[1], int(l), width])
            start_point = [0, int(l)]
        parts.append([start_point[0], start_point[1], height - start_point[1], width])
    else:
        parts = [[0, 0, height, width], ]
    return parts


###################deal3新增#################################################

def deal_y_half_no_rec(img_path):
    # 基于纵坐标对图片进行分割【主要是处理局部分割线的情况】
    original_img, height, width, gray = load_image(img_path)
    blurred = Gaussian_Blur(gray)
    gradient_img = Sobel_gradient(blurred)
    thresh_blue_img = Thresh_and_blur(gradient_img)
    thresh_blue_img = doErode(doDilate(thresh_blue_img, 3), 2)
    parts = find_parts(original_img.copy(), thresh_blue_img, height, width)
    pics = []
    change_info = collections.defaultdict(dict)
    for idx, part in enumerate(parts):
        apic = crop_image(original_img, *part)
        pics.append(apic)
        change_info[idx]["change"] = part
    return original_img, pics, change_info


def deal_for_y_parts_no_rec(change_info, pic):
    height, width = pic.shape[:2]
    gray = cv2.cvtColor(pic.copy(), cv2.COLOR_BGR2GRAY)
    blurred = Gaussian_Blur(gray)
    gradient_img = Sobel_gradient(blurred)
    thresh_blue_img = Thresh_and_blur(gradient_img)
    max_line_p = find_longest_line(thresh_blue_img, height, width)
    if max_line_p:
        pic, change_info = crop_half_with_change_info(change_info, pic, height, width, max_line_p)
    else:
        change_info[1] = {}
        change_info[1]["change"] = [0, 0, height, width]
        print("not line")
    return pic, change_info


def crop_half_with_change_info(change_info, original_img, height, width, max_line_p):
    # 根据分割线对原图进行分割并合并
    x1, y1, x2, y2 = max_line_p
    x, y = loc_point(height, (x1, y1), (x2, y2))
    pic1 = crop_image(original_img, 0, 0, height, y)
    change_info[1] = {}
    change_info[1]["change"] = (0, 0, height, y)
    pic2 = crop_image(original_img, x, 0, height, width - x)
    change_loc = (pic1.shape[0], 0)
    change_info[2] = {}
    change_info[2]["change"] = (x, 0, height, width - x, change_loc)
    pic2 = cv2.resize(pic2, tuple(pic1.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)
    res = pic_add(pic1, pic2)
    # res = imgEnhance(res, a=0.9, b=0)
    return res, change_info


def to_standard_pic(pic, standard_width):
    # 更改图片的形状至标准的宽度，等比的缩放
    return cv2.resize(pic, (standard_width, int(pic.shape[0] * (standard_width / pic.shape[1]))),
                      interpolation=cv2.INTER_CUBIC)


def pic_add(pic1, pic2):
    # 两张图片进行纵向拼接
    return np.vstack((pic1, pic2))


def mix_pics(pics, change_info, standard_width):
    # 组合图片成为大长图
    opic = pics[0]
    pic = to_standard_pic(opic, standard_width)
    change_info[0]["percent_change"] = ((0, 0), *opic.shape[:2], *pic.shape[:2])
    for idx, apic in enumerate(pics[1:]):
        p = to_standard_pic(apic, standard_width)
        change_info[idx + 1]["percent_change"] = ((0, 0), *(apic.shape[:2]), *(p.shape[:2]))
        height, width = p.shape[:2]
        change_info[idx + 1]["change2"] = (0, 0, height, width, (pic.shape[0], 0))
        pic = pic_add(pic, p)
    return pic, change_info


################################################################################################

def deal3(img_path, standard_width=900):
    # standard_width是设置的标准宽度，具体值依照产品给出的标准。这边直接先设置900
    # 根据行【如果有这样的行】进行切割
    original_img, pics, change_info = deal_y_half_no_rec(img_path)
    # change_info 这个变量很重要，保存了图片在所有步骤中经历的移行变位的操作记录，会跟随后续的步骤时刻记录。
    # 创建存储上一步切割后图片进行各种处理后拼接而成的图
    l_pics = []
    for idx, pic in enumerate(pics):
        #
        loc, change_info_ = deal_for_y_parts_no_rec(change_info[idx], pic)
        change_info[idx] = change_info_
        l_pics.append(loc)
    # 返回了大长图，以及原始图片变成这张图需要经历的所有操作。
    pic, change_info = mix_pics(l_pics, change_info, standard_width)
    return original_img, pic, change_info


if __name__ == '__main__':
    imgs = ["../data/pic/bobo_IMG_9927.JPG", "../data/pic/bobo_IMG_1879.JPG", "../data/pic/bobo_IMG_1880.JPG",
            "../data/se_pic/math_IMG_1890.JPG"]
    for img in imgs:
        deal3(img)
