#!/usr/bin/python
# coding=utf-8
import re

import web

from answer_recognition import ans,load_img

urls = (
    '/rec', "QuestionRec"
)

app = web.application(urls, globals())
A = ans()

class QuestionRec(object):
    def POST(self):
        i = web.input()
        x = i["file"]
        pics = [load_img(j) for j in x]
        return A.see_a_pic3(pics)


if __name__ == "__main__":
    app.run()
