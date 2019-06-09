import json

import cv2
import time
import os
import numpy as np
import tensorflow as tf
from box_deal_functions import load_labelmap,convert_label_map_to_categories,create_category_index


def change_size(path, ifChange):
    # 载入图片
    # 如果更改图片大小，则更改图片的尺寸至 （900,1200）
    img = cv2.imread(path)
    if ifChange:
        return cv2.resize(img, (900, 1200), interpolation=cv2.INTER_CUBIC)
    else:
        return img


def load_image(path, ifChange=True):
    # 获取图片，长， 宽，灰度图
    img = change_size(path, ifChange)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    return img, height, width, gray


def str2json(aline):
    # 对文本进行处理，返回可以json格式化的文本
    return aline.strip(). \
        replace("\\", "").replace("\"[", "[").replace("]\"", "]").replace("\"{", "{").replace("}\"", "}")


def load_img(res):
    return cv2.resize(cv2.imdecode(np.asarray(bytearray(res), dtype="uint8"), cv2.IMREAD_COLOR), (1200, 1200),
                      interpolation=cv2.INTER_CUBIC)


class ans:
    def __init__(self):
        self.category_index = self.init_cata()
        self.start()

    def start(self):
        ckpt_path = os.path.join("inference_graph", 'frozen_inference_graph.pb')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            config = tf.ConfigProto(device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
                                    inter_op_parallelism_threads=2,
                                    intra_op_parallelism_threads=4,
                                    log_device_placement=True)
            self.sess = tf.Session(graph=detection_graph, config=config)
            print(self.sess.list_devices())
        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def init_cata(self):
        label_map = load_labelmap("../data/labelmap.pbtxt")
        categories = convert_label_map_to_categories(label_map, max_num_classes=4, use_display_name=True)
        return create_category_index(categories)

    def to_point(self, height, width, point):
        return (int(round(height * point[0])), int(round(width * point[1])))

    def see_a_pic3(self, images, limit=0.7):
        t1 = time.time()
        BATCH_SIZE = 5
        f_lis = []
        for i in range(0, len(images), BATCH_SIZE):
            t2 = time.time()
            lis = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: images[i:i + BATCH_SIZE]})
            print("run_time:", time.time() - t2)
            return lis
        print("total time:", time.time() - t1)
        return f_lis