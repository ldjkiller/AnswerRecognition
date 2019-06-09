# coding:utf8
from config import api_id, api_key, api_pwd
from aip import AipOcr

""" 如果有可选参数 """


class bns:
    def __init__(self):
        self.options = self._init_options()
        self.client = AipOcr(api_id, api_key, api_pwd)

    def _init_options(self):
        options = {}
        options["language_type"] = "CHN_ENG"
        options["detect_direction"] = "true"
        options["detect_language"] = "true"
        options["probability"] = "true"
        return options

    def ana(self, img_file):
        return self.client.general(img_file, self.options)


if __name__ == '__main__':
    from pic_rec.pic_deal import load_image

    img, _, _, _ = load_image("../data/se_pic/chi_IMG_1884.JPG")
    print(img)
    b = bns()
    res = b.ana(img)
    print(res)
