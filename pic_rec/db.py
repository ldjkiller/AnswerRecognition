import editdistance


# 创建db类，closest方法判定库中跟目标字符编辑距离最近的字符

class db:
    def __init__(self, data):
        self.data = data

    def closest(self, item):
        mi = 100
        minK = None
        for k, v in self.data.items():
            v = v[:30]
            res = editdistance.eval(v, item)
            lis = (len(item), len(v))
            dis = abs(len(item) - len(v))
            num = (res - dis) / min(lis) + dis / max(lis)
            if num < mi:
                mi = num
                minK = k
        if mi < 0.7:
            # print(self.data[minK],mi)
            return minK, mi

    def closest_new(self, item_lis):
        mi = 100
        minK = None
        old, now = item_lis
        for k, v in self.data.items():
            ln = len(now)
            lv = len(v)
            if lv < 25 and ln > 6:
                if len(old) <= 5:
                    v = v[:5+ln]
                else:
                    v = v[:ln]
            else:
                continue
            # old_res = editdistance.eval(v,old+now)
            now_res = editdistance.eval(v, now)
            lis = (ln, lv)
            dis = abs(ln - lv)
            # score_bias = -0.2 if old_res >= now_res else 0.2
            num = (now_res - dis) / min(lis) + dis / max(lis)
            if num < mi:
                mi = num
                minK = k
        print("now", now)
        if minK:
            print(self.data[minK], mi)
        print("###################")
        if mi < 0.6:
            # print(self.data[minK],mi)
            return minK, mi


# 模拟数据库
data = {
    "math": {
        101: "一元二次方程ax2+bx+c=0(a≠0)的根的",
        102: "关于x的一元二次方程kx2+2x",
        103: "一元二次方程x2-2x=0的根的判别式的值是",
        104: "方程2x2-4x+1=0中,=b2-4ac=",
        105: "方程x2+mx-1=0的根的判别式的值为8,则m的值是",
        114: "方程x2-7x-2=0的根的情况为",
        115: "方程42-2+4=0的根的情况是",
        116: "关于x的方程x2+ax-1=0的根的情况是",
        117: "a,b,c为常数,p(a,c)在第二象限,则关于x的方程ax2+bx+c=0的根的情况是",
        118: "不解方程,判断下列一元二次方程的根的情况",
        119: "一元二次方程x2+23x+m=0有两个不相等的实数根,则",
        106: '方程x2-2x+k=0有两个相等的实根,则k的值是',
        # 7: '如上页图,在正方形铁皮上剪下一个圆形和一个扇形,使之恰好围成一个圆锥模型',
        # 8: '四边形ABCD内接于,A:C=1:3,则A=',
        # 9: '如图，四边形ABCD内接于，AB是直径，过C的切线与AB的延长线交于P',
        # 10: '圆锥的底面半径为3cm,母线长为5cm,则它的侧面积是',
        # 11: '如图,矩形ABCD与圆心在AB上的交于点G,B,F,E,CB=8cm,AG=1cm,',
        # 12: '如图,已知四边形ABCD是边长为2的菱形,点E,B,C,F都在以D为圆心的同',
        # 13: '如图,的外切正六边形ABCDEF的边长为2,则图中阴影部分的面积为',
        # 20: "已知A(1,-1),B(2,0.5),C(-2,3),D(-1,-3),E(0,-3),F(4,-1.5),G(5,0),其中在第四象",
    },
    "eng": {
        201: '语块积累',
        202: '根据所给的句子意思，将打乱的字母组合成单词，再将单词填入相应的句子',
        203: '按字母表的顺序以及所给字母写出下列字母的左邻右舍',
        204: '语音知识',
        205: '根据所给的表单符号，连词成句',
        206: '对话配对',
        207: '按首字母的顺序将下列人名排序,并抄写在四线格内。',
        208: '选择填空',
        209: '看图填入适当的单词。',
        214: '补全对话',
        215: '书面表达',
    },
    "sci": {
        301: "地球表面昼夜交替,主要是因为地球的",
        302: "小明从北京去新疆乌鲁木齐旅游,在北京登机时太阳刚",
        303: "下图中,正确示意地球自转方向的是",
        304: "我们每天看到日月星辰东升西落的现象是因为",
        305: "如图太阳光照图中,阴影部分表示黑",
        306: "关于晨昏线的说法正确的是",
        307: "下列现象中,与地球自转有关的是",
        308: "演示地球自转时,用手电简模拟“太阳”,拨动地球仪(如",
        309: "读地球自转示意图,回答下列问题",
        310: "地球自转的方向是自西向东,但从不同角度观察,这个",
        311: "下列现象:①日月星展的东升西落;②地球上有四季更",
        312: "下列有关地球自转的说法,正确的是",
        313: "如图为“歼15”飞机首次从“辽宁号”航母上起飞时的照",
        314: "游泳运动员在游泳过程中,下列说法不正确的是”,",
        315: "公安部门要求小型客车的驾驶员和前排乘客必须使用",
        316: "将弹簧测力计右端固定,左端与木块相连,木块放在上",
        317: "如图所示,一个空的料药瓶,瓶口扎上橡皮膜,竖直浸",
        318: "甲、乙两辆汽车在同一条平直公路上行驶,甲车中乘客",
        319: "贴在竖直墙面上的塑料吸盘挂钩(塑料吸盘挂钩重力不",
        320: "王明同学早晨锻身体时，先沿较光滑的竖直杆匀速向",
        321: "如图所示是探究“阻力对物体运动影响”的实验装置,下",
        322: "某段水平公路路面所能承受的最大压强是8×105帕",
        323: "“漂移”是一种高难度的汽车驾驶技巧,有一种“漂移",
        # 24: ""
    },
    "chi": {
        1: "黄河是中华民族的摇篮，它记载着源远流",
        2: "下列加点字注音全部正确的一项是",
        3: "阅读下面的文段，根据拼音写汉字。",
        4: "下列加点词，结合语境解释有误的一项是",
        5: "依次填入下列横线上的词语，最恰当的一项是",
        6: "指出下列各句运用的描写方法。",
        7: "下列句子没有使用修辞手法的一项是",
        8: "下列居中标点使用正确的一项是",
        9: "下列各组词语中加点字注音无误的一项是",
        10: "下列各组词语书写完全正确的一项是",
        11: "下列句子中加点的成语使用不恰",
        12: "依次填入下面这段文字线处的语句,衔接最恰当的",
        13: "下列句中没有语病、句意明确的一项是",
        14: "仿照画波浪线的句子,在横线上续写内容,使之构成排",
        15: "黄河是中华民族的摇篮，它记载着源远流",
        16: "为继承和弘扬中华优秀传绕文化,",
    }
}



# data = {
#     "math": {
#         1: "若是方程的一个解,且,则的符号是",
#         2: "若二元一次方程有正整数解,则的值为",
#         3: "方程组x+y=7,xy=12的一个解是",
#         4: "若二元一次方程组的解中y=0,则a:b",
#         5: "已知方程3x+y=12有很多解,请你随意写出互为相反数的一组解",
#         14: "已知x-5y=-5,则5-x+5y的值是",
#         15: "若是关于ab的二元一次方程ax+ay-b=7的一个解,则代数式(x+y)-1的值是",
#         16: "已知二元一次方程,用含y的代数式表示x,则x=",
#         17: "已知方程是关于x、y的二元一次方程,求m、n的值",
#         18: "若是关于x,y的二元一次方程3x-y+a=0的一个解,求a的值",
#         19: "小华不小心将墨水溅在同桌小丽的作业本上,结果二元一次方程组中第一个方程y",
#         6: '如上页图,过外一点P引的两条切线PA,PB,切点分别是A,B,OP交',
#         7: '如上页图,在正方形铁皮上剪下一个圆形和一个扇形,使之恰好围成一个圆锥模型',
#         8: '四边形ABCD内接于,A:C=1:3,则A=',
#         9: '如图，四边形ABCD内接于，AB是直径，过C的切线与AB的延长线交于P',
#         10: '圆锥的底面半径为3cm,母线长为5cm,则它的侧面积是',
#         11: '如图,矩形ABCD与圆心在AB上的交于点G,B,F,E,CB=8cm,AG=1cm,',
#         12: '如图,已知四边形ABCD是边长为2的菱形,点E,B,C,F都在以D为圆心的同',
#         13: '如图,的外切正六边形ABCDEF的边长为2,则图中阴影部分的面积为',
#         20: "已知A(1,-1),B(2,0.5),C(-2,3),D(-1,-3),E(0,-3),F(4,-1.5),G(5,0),其中在第四象",
#         21: "在平面直角坐标系中,点一定在",
#         22: "已知点A(3a,2b)在x轴上方,y轴的左边,则点A到x轴,y轴的距离分别为",
#         23: "点P(a,b)满足,则点P的坐标为",
#         24: "在坐标平面内,有一点P(a,b),若ab=0,则P点的位置在",
#         25: "已知点A(2,-2),如果点A关于x轴的对称点是B,点B关于原点的对称点是C,那么C点的坐",
#         26: "将点P(-4,3)先向左平移2个单位,再向下平移2个单位得点P,则点P的坐标为",
#         27: "已知A(5,8),B(-8,8),过这两点作直线AB,则AB",
#         28: "如右图所示,半圆AB平移到半圆CD的位置时所扫过的面积为",
#         29: "已知:△ABC的顶点坐标分别为A(-4,-3),B(0,-3),C(-2,1),如",
#         30: "在第二象限的M点,到x轴和y轴的距离分别是8和5,那么点M的坐标为",
#         31: "已知:点P的坐标是(m,-1),且点P关于x轴对称的点的坐标是(-3,2n),则m",
#         32: "线段AB的端点A的坐标是(2,3),点B的坐标是(5,2),现将线段AB平移至线段A'B',如果A",
#         33: "已知A(-2,0),B(a,0)且AB=5,则B点坐标为",
#         34: "点K在第三象限,且横坐标与纵坐标的积为8,写出两个符合条件的点",
#         # 35:"",
#         # 36:"",
#         # 37:"",
#         # 38:"",
#         # 39:"",
#     },
#     "eng": {
#         1: '语块积累',
#         2: '根据所给的句子意思，将打乱的字母组合成单词，再将单词填入相应的句子',
#         3: '按字母表的顺序以及所给字母写出下列字母的左邻右舍',
#         4: '语音知识',
#         5: '根据所给的表单符号，连词成句',
#         # 6:'11.Is she Mrs. Green?',
#         # 7:'12.Whats',
#         8: '选择填空',
#         14: '补全对话',
#         15: '书面表达',
#     },
#     "sci": {
#         1: "如图是人的生长曲线图，以下关于人生长发",
#         2: "人出生后，还要经历婴幼儿期、少年",
#         3: "以下关于青春期生理变化的叙",
#         4: "(2017・福建)下列属于青春期发育特",
#         5: "如图为科研小组所测定的某地男生、女生",
#         6: "牛顿第一定律适合于",
#         7: "根据牛顿第一定律可知",
#         8: "让一个表面光滑的小钢球在没有任何阻力的水平面上",
#         9: "关于力和运动的关系,下列说法正确的是",
#         10: "如图所示,小物块A和弹簧放在光滑的水平面上,弹簧",
#         11: "如图所示,小球沿弧形斜槽从A点运动到水平轨道的B",
#         12: "原来悬挂着的小球,线断后下落的遠度越来越快是由",
#         13: "在学习“物体运动状态改变的原因”时,老师做了如图的",
#         14: "宇航员在“天宫一号”中为我们演示了“太空中的单摆”,",
#         15: "在“探究牛顿第一定律”的实验中：",
#         16: "在水平木板上铺上粗糙程度不同的材料,小车自斜面",
#         17: "一个物体在两个力的作用下,如果保持",
#         18: "二力平衡的条件是:两个力要作用在",
#         19: "一个物体在平衡力的作用下,将保持",
#         20: "重力为10牛的物体静止在水平桌面上,物体受到",
#         21: "一个物体只受到两个力的作用,这两个力的三要素完全",
#         22: "下列情况中,不属于二力平衡的是",
#         23: "物体在平衡力的作用下,下列说法中哪个正确",
#         24: "如图所示,使纸片扭转一定角度,放手后纸片不能保持",
#         25: "一本书放在水平桌面上,下列各对力中属",
#         26: "当一辆白行车在水平地面上做匀速直线运动时,这辆自",
#         27: "如图所示,用线检着的小球在",
#         28: "如图是“探究二力平衡的条件”的实验装置。",
#         29: "日常生活和生产中,摩擦无时不在。下列现象属于减小",
#         30: "起重机吊着货物时,货物所受重力G和拉力F之间的关",
#         31: "如图所示,甲、乙为两容器,用一带阀门的管子相连,两",
#         32: "如图,关于a、b、c、d四点处液体产生的强大小的比较,",
#         33: "如图所示,小车被人推开后向前运动,最终停下了。对",
#         34: "如图所示,木块a放在粗糙水平桌面上,木块b放在木块",
#         35: "如图所示,甲、乙两物体在水平桌面上做匀速直线运动",
#         36: "在“探究液体内部的压强与哪些因素有关”实验中,各小",
#         37: "探究滑动摩擦力的大小与什么因素有关的实验中,做了",
#         38: "如图所示,在一只透明的饮",
#         # 30: "",
#         # 30: "",
#         # 30: "",
#         # 30: "",
#         # 30: "",
#     },
#     "chi": {
#         1: "黄河是中华民族的摇篮，它记载着源远流",
#         2: "下列加点字注音全部正确的一项是",
#         3: "阅读下面的文段，根据拼音写汉字。",
#         4: "下列加点词，结合语境解释有误的一项是",
#         5: "依次填入下列横线上的词语，最恰当的一项是",
#         6: "指出下列各句运用的描写方法。",
#         7: "下列句子没有使用修辞手法的一项是",
#         8: "下列居中标点使用正确的一项是",
#         9: "下列各组词语中加点字注音无误的一项是",
#         10: "下列各组词语书写完全正确的一项是",
#         11: "下列句子中加点的成语使用不恰",
#         12: "依次填入下面这段文字线处的语句,衔接最恰当的",
#         13: "下列句中没有语病、句意明确的一项是",
#         14: "仿照画波浪线的句子,在横线上续写内容,使之构成排",
#         15: "黄河是中华民族的摇篮，它记载着源远流",
#         16: "为继承和弘扬中华优秀传绕文化,",
#     }
# }
