# AnswerRecognition
主要的代码包括两部分功能，一个功能是题目切割，第二个功能是答案识别。

## 答案识别
aa文件夹是整个答案识别的服务，到 [*这里*](https://pan.baidu.com/s/1Gafj6qaaicOnmiEWyqaTXQ) 密码：d83d  
下载我已经训练好的模型，解压后的inference_graph文件夹放到aa文件夹下面。
运行
```
python3 my_server.py
```

## 题目分割
注意，这个需要先在百度去申请一个通用文字识别含位置信息版，这个接口的作用是为了粗粒度地定位图片中的文字，为后续的题目分割做准备。申请到的信息填在config.py文件中。
运行 pic 文件夹下的pic_rec_main.py，会对data文件夹下的部分图片进行测试


提示：最好选用带gpu的机子，速度杠杠的。

## 效果展示

<div align=center><img src="https://github.com/ldjkiller/AnswerRecognition/blob/master/result_pic/22DEF5C3-6BFC-489c-AADD-1A78F3FEA27A.png" width="400" height="600" alt="图片加载失败时，显示这段字"/></div>

<div align=center><img src="https://github.com/ldjkiller/AnswerRecognition/blob/master/result_pic/23FF8751-260F-428e-A4C0-59E01861082A.png" width="400" height="600" alt="图片加载失败时，显示这段字"/></div>
