# coding:utf8
import collections
import json
import re
#
import cv2
import requests
from requests_toolbelt import MultipartEncoder

from db import db as question_db
from db import data
from pic_deal import deal3
from blurDetector import BlurDetector
from baiduApi import bns
import numpy as np

A = "http://localhost:8080/rec"
B = bns()


def draw_box(original_img, box, t=0):
    # 切割【基于边缘坐标处理后进行切割】
    x, y = [], []
    for i in box:
        x.append(i[0])
        y.append(i[1])
    minx, miny = min(x) - t, min(y) - t
    height = max(y) - miny + t
    width = max(x) - minx + t
    return crop_image(original_img, minx, miny, height, width)


def crop_image(img, minx, miny, height, width):
    # 切割【根据左上坐标和长宽数据进行分割】
    return img[miny:miny + height, minx:minx + width]


def get_file_content(filePath):
    # baidu_文本识别api文件读入
    with open(filePath, 'rb') as fp:
        return fp.read()


def deal_lis(lis, answer_map):
    # 计算题目与题目之间的距离，并生成坐标
    start = lis[0]
    res = []
    answer_lis = []
    for v in lis[1:]:
        y, x, h, width = start
        y2, x2, h2, width2 = v
        answers = [answer_map[i] for i in range(x, x2, 1) if i in answer_map]
        start = v
        # 小于0的数都为0
        res.append([y, x - 15 if x - 15 > 0 else 0, x2 - x + 30, width])
        answer_lis.append(answers)
    return res, answer_lis


def deal_dic(adic):
    # 找到最匹配的字符【单套题库内】
    new_dic = {}
    new_dic2 = {}
    for key, dic in adic.items():
        min_k = dic["mink"]
        min_v = dic["minV"]
        if min_k not in new_dic or (min_k in new_dic and new_dic[min_k] > min_v):
            new_dic[min_k] = min_v
            new_dic2[min_k] = key
    key_set = set(new_dic2.values())
    return [v["lis"] for i, v in sorted(adic.items(), key=lambda x: x[0]) if i in key_set]


def load_data(name):
    # 从库里面获取数据
    questions = {}
    if "math" in name:
        maths = data["math"]
        questions.update(maths)
    if "eng" in name:
        engs = data["eng"]
        questions.update(engs)
    if "sci" in name:
        scis = data["sci"]
        questions.update(scis)
    if "chi" in name:
        chis = data["chi"]
        questions.update(chis)
    return questions


def charge_blur(pic):
    # 判定图片是否模糊
    bd = BlurDetector(pic)
    info = bd.Test_Tenengrad()
    if not info:
        raise IOError("WRONG PIC")


def rec_question_with_location(pic, width, db):
    res = B.ana(pic)
    adic = collections.defaultdict(dict)
    for idx, word_info in enumerate(res["words_result"]):
        word = word_info["words"]
        loc = word_info["location"]
        # print(word)
        closest = db.closest(word)
        if closest:
            mink, mi = closest
            x, y, w, h = loc["top"], loc["left"], loc["width"], loc["height"]
            vlis = [0, x, h, width]
            adic[idx]["lis"] = vlis
            adic[idx]["mink"] = mink
            adic[idx]["minV"] = mi
    if adic:
        return adic
    else:
        raise IOError("WRONG PIC")


def ori_location_2_new_location(answer_map, process_info):
    # print("0", answer_map)
    new_dic = {}
    for id in process_info:
        info = process_info[id]["change"]
        answer_map_new = locat_new(answer_map, *info)
        new_dic[id] = answer_map_new
    answer_map = new_dic.copy() if new_dic else answer_map
    # print("1", answer_map)
    new_dic = {}
    for id in process_info:
        new_dic2 = {}
        for idx in process_info[id]:
            if type(idx) == int:
                answer_map_new = locat_new(answer_map[id], *process_info[id][idx]["change"])
                new_dic2.update(answer_map_new)
        new_dic[id] = new_dic2 if new_dic2 else answer_map[id]
    answer_map = new_dic.copy() if new_dic else answer_map
    # print("2", answer_map)
    new_dic = {}
    for id in process_info:
        new_dic2 = {}
        if "percent_change" in process_info[id]:
            info = process_info[id]["percent_change"]
            answer_map_new = percent_change(answer_map[id], *info)
            new_dic2.update(answer_map_new)
        new_dic[id] = new_dic2 if new_dic2 else answer_map[id]
    answer_map = new_dic.copy() if new_dic else answer_map
    # print("3", answer_map)
    new_dic = {}
    for id in process_info:
        new_dic2 = {}
        if "change2" in process_info[id]:
            info = process_info[id]["change2"]
            answer_map_new = locat_new(answer_map[id], *info)
            new_dic2.update(answer_map_new)
        new_dic[id] = new_dic2 if new_dic2 else answer_map[id]
    answer_map = new_dic.copy() if new_dic else answer_map
    # print("4", answer_map)
    new_dic = {}
    for id in answer_map:
        for i in answer_map[id]:
            py, px = answer_map[id][i]["point"]
            new_dic[int(py)] = answer_map[id][i]["tag"]
    # print(new_dic)
    return new_dic


def locat_new(answer_map, minx, miny, height, width, change_loc=()):
    # 定位新的位置
    new_answer_map = collections.defaultdict(dict)
    for i in answer_map:
        y, x = answer_map[i]["point"]
        if minx < x < minx + width and miny < y < miny + height:
            new_answer_map[i]["tag"] = answer_map[i]["tag"]
            new_x = x - minx
            new_y = y - miny
            if change_loc:
                new_x += change_loc[1]
                new_y += change_loc[0]
            new_answer_map[i]["point"] = (new_y, new_x)
    return new_answer_map


def percent_change(answer_map, old_lu_p, old_h, old_w, new_h, new_w):
    # 尺寸变化的新位置定位
    new_answer_map = collections.defaultdict(dict)
    minx, miny = old_lu_p
    for i in answer_map:
        y, x = answer_map[i]["point"]
        if minx < x < minx + old_w and miny < y < miny + old_h:
            new_answer_map[i]["tag"] = answer_map[i]["tag"]
            new_x = float(x / old_w) * new_w
            new_y = float(y / old_h) * new_h
            new_answer_map[i]["point"] = (new_y, new_x)
    return new_answer_map


def load_img(res):
    return cv2.resize(cv2.imdecode(np.asarray(bytearray(res), dtype="uint8"), cv2.IMREAD_COLOR), (1200, 1200),
                      interpolation=cv2.INTER_CUBIC)


def mix_change(all_pic):
    # 整合所有的图片，构建大长图，并基于大长图重新定位题目切割位置和答案识别的位置
    lis2, a_map = [], {}
    old_h = 0
    start_pic = None
    for id, pic_info in sorted(all_pic.items(), key=lambda x: x[0]):
        h = pic_info["pic_h"]
        pic = pic_info["pic"]
        lis = pic_info["lis"]
        answer_map = pic_info['answer_map']
        if id == 0:
            start_pic = pic.copy()
        else:
            start_pic = np.vstack((start_pic, pic))
        for loc in lis:
            lis2.append([loc[0], loc[1] + old_h, loc[2], loc[3]])
        for h_loc in answer_map:
            a_map[h_loc + old_h] = answer_map[h_loc]
        old_h += h
    return start_pic, lis2, a_map


def to_rb_img(res):
    # CV2的img转二进制流
    res2 = cv2.imencode('.jpg', res)[1]
    data_encode = np.array(res2)
    return data_encode.tostring()

def post_info(all_files):
    # 图片答案识别接口调用
    header = {}
    file_dic = {}
    for idx, f in enumerate(all_files):
        file_dic["img_%s" % idx] = ("img[%s]" % idx, open(f, "rb").read())
    encode_data = MultipartEncoder(file_dic)
    header['Content-Type'] = encode_data.content_type
    res = requests.post(A, headers=header, data=encode_data)
    return res.content


def deal_answer_list(answer_lis):
    # 图片识别接口返回值处理，处理成一张一张的
    adic = {}
    for idx, answer in enumerate(answer_lis):
        adic[idx] = {idxx: {"point": (int((i[0][0] + i[1][0]) / 2), int((i[0][1] + i[1][1]) / 2)), "tag": i[2]} for
                     idxx, i in enumerate(answer)}
    return adic


def start(o_pics, subject):
    # 载入数据
    local_data = load_data(subject)
    db = question_db(local_data)
    all_pic = collections.defaultdict(dict)
    for idx, o_pic in enumerate(o_pics):
        # 判断是否模糊
        charge_blur(o_pic)
        s = get_file_content(o_pic)
        img = load_img(s)
        # 处理图片
        o_pic, pic, process_info = deal3(img)
        h, w = pic.shape[:2]
        # 文本识别+判断图片的可用性
        adic = rec_question_with_location(to_rb_img(pic), w, db)
        lis = deal_dic(adic)
        all_pic[idx]["pic_h"] = h
        all_pic[idx]["pic"] = pic
        all_pic[idx]["lis"] = lis
        all_pic[idx]["answer_map_process"] = process_info
    # 答案识别
    res = post_info(o_pics).decode()
    res = json.loads(res)
    # res = [i for i in res]
    answer_dic = deal_answer_list(eval(post_info(o_pics).decode()))
    for i in answer_dic:
        new_answer_map = ori_location_2_new_location(answer_dic[i], all_pic[i]["answer_map_process"])
        all_pic[i]["answer_map"] = new_answer_map

    # 遍历所有的截图位置信息进行截图
    mix_img, lis, a_map = mix_change(all_pic)
    # 末尾补上下沿位置信息
    lis.append([0, mix_img.shape[0], 0, 0])
    lis2, answer_lis = deal_lis(lis, a_map)
    # 遍历所有的截图位置信息进行截图
    for idx, v in enumerate(lis2):
        y, x, h, width = v
        # 切题目、且使用已经识别了答案的图片
        res = crop_image(mix_img, y, x, h, width)
        print("answer:", answer_lis[idx])
        cv2.imshow("res", res)
        cv2.waitKey(2500)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pic2 = "../data/se_pic/IMG_1947.JPG"
    pic3 = "../data/se_pic/chi_IMG_1884.JPG"
    pic4 = "../data/se_pic/chi_IMG_1885.JPG"
    pic5 = "../data/se_pic/eng_IMG_1888.JPG"
    pic6 = "../data/se_pic/eng_IMG_1889.JPG"
    pic7 = "../data/se_pic/math_IMG_1890.JPG"
    pic8 = "../data/se_pic/sci_IMG_1886.JPG"
    pic9 = "../data/se_pic/sci_IMG_1887.JPG"
    lis = [pic4, pic5, pic6, pic7, pic8, pic9]
    start([pic3, pic4], "chi，math")
