import cv2
import numpy as np
from skimage import filters


# 检测图片的可用性【是否模糊】

class BlurDetector:
    def __init__(self, img):
        self.img = img

    def Test_Tenengrad(self):
        return self._Tenengrad()

    def preImgOps(self):
        # 图片预处理
        img = cv2.imread(self.img)
        # img = self.img
        reImg = cv2.resize(img, (1200, 1200), interpolation=cv2.INTER_CUBIC)
        img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]
        return img2gray, height, width, reImg

    def _imageToMatrix(self, image):
        # 图片转换成matrix
        imgMat = np.matrix(image)
        return imgMat

    def _Tenengrad(self):
        img2gray, height, width, reImg = self.preImgOps()
        f = self._imageToMatrix(img2gray)
        tmp = filters.sobel(f)
        source = np.sum(tmp ** 2)
        source = np.sqrt(source)
        if source > 60:
            return True
        else:
            cv2.imshow("o_img", reImg)
            cv2.waitKey(1000)
            print("图片过于模糊，请重新拍摄")
