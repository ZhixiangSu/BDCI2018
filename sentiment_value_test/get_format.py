# coding=gbk
import pandas as pd
import torch
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
raw = pd.read_csv("../train.csv",encoding="UTF-8")
subject_dic = {"价格": 1, "内饰": 2, "配置": 3, "安全性": 4, "外观": 5, "操控": 6, "油耗": 7, "空间": 8, "舒适性": 9, "动力": 10}
standard_sub = ["价格", "内饰", "配置", "安全性", "外观", "操控", "油耗", "空间", "舒适性", "动力"]
key_words=["家庭","恭喜","报价","上下班","塑料","轴承","耐看","减震","优势","5000","刹车","排量","刹车灯","安全性","100","够用","担心","超车","宽敞","太小","优惠","图像","安卓","静音","里程","东西","17","公路","2019","循环","噪音","不好","手刹","刹车油","推荐","胶套","cx5","价格","玻璃","指导价","自由","合适","cd","发电机","影响","空调","车尾","实用","车漆","维修","gps","后期","80","立德","平均速度","感受","急刹车","匀速","垃圾","文件","机油","熄火","加油","后轮","操控性","气囊","一分钱","清零","舒适","动力","好看","尖叫","味道","销售","市内","落地","喷漆","碰撞","用料","测试","毛病","地区","不响","升级","靠背","马力","支持","内饰","操控","加速","车主","汽车","难看","更新","实时","开关","下坡","经销商","大气","产品","价钱","停车","全时","建议","后排","风噪","贷款","异响","北京","关税","银色","油表","发出","楼兰","足够","指南","14","上市","人比","启动","刹车片","彩屏","外观","减速","很大","路况","做工","原车","汽贸","纠结","音响","材料","变化","加满","记忆","性价比","颜色","关闭","abs","积碳","fb","制动","不烧","2w","百公里","改改","11","润滑","销量","裸车","界面","费用","座椅","40","空间","长途","庞大","添加剂","国内","刹车盘","雷达","车身","配置","电动","国道","设计师","宝马","消耗","后视镜","菲罗多","关系","外形","换代","车况","爆震","漂亮","高德","加装","12","人大","前排","压缩机","全款","ej","轿车","辅助","内部空间","实惠","蓝牙","不到","不用","胎压","综合","城区","速度","儿子","优点","底盘","外观设计","中控","尾门","gla","还好","30","轴距","全系","电子","真心","13","每次","省油","效果","漏油","价位","地图","细节","油耗","希望","有没有","清洗","0w20","车价","尊贵","主机","山寨","老款","档位","力狮","家用","助力","卡钳","连接","对置","低配","情况","es","发现","库存","镀晶","瞬时","usb","角度","手续费","q5","16","昂科威","冰雪","尾灯","后备箱","控制","小熊","国产","城市","车子","曼牌","配件","wd40","满意","方向盘","差速器","音质","天窗","级别","国产车","真皮","镀铬","确实","原厂","19","大灯","一代","利润","20","摄像头","不在意","车衣","视野","面板","制冷","cvt","美孚","打蜡","鹰眼","避震","前脸","只能","轮胎","影像","力度","发动机","没换","比傲","降价","rav4","两个","便宜","声音","解决","高配","cx","耐脏","变速箱","价格便宜","导航","刹车踏板","开过","乘坐","奇骏","xt","堵车","黑色","个油","均速","估计","越改越","空滤","舒适性","獠牙","主动","更好","定速","工作","性能","dixcel","护板","现金",
]
temp=pd.read_csv("../tf_idf.csv",encoding="gbk")
words_tfidf = temp[key_words]
values = [[] for i in range(len(raw))]
sp_content=[[] for i in range(len(raw))]
sp_word=[]
for i in range(len(raw)):
    for j in range(1, len(subject_dic) + 1):
        v = 0
        max_tfidf=0
        max_word=""
        words = jieba.lcut(raw["content"].values[i])
        for word in words:
            if (word in words_tfidf.columns.tolist()):
                if(words_tfidf[word].values[j]>max_tfidf):
                    max_tfidf=words_tfidf[word].values[j]
                    max_word=word
                v += words_tfidf[word].values[j]
        values[i].append(v)
        sentences=re.split("[，。！？?]",raw["content"].values[i])
        sp_word.append(max_word)
        for sentence in sentences:
            if(sentence.count(max_word)>0):
                sp_content[i].append(sentence)
                break
values = pd.DataFrame(values)
sp_content=pd.DataFrame(sp_content)
values.columns = standard_sub
# values["subject"]=raw["subject"]
tf_idf=values
for i in range(len(tf_idf)):
    sum = 0
    for j in range(len(tf_idf.values[i])):
        sum += tf_idf.values[i][j]
    for j in range(len(tf_idf.values[i])):
        tf_idf.values[i][j] /= sum
threshold=0.19
result_tfidf=[]
for i in range(len(tf_idf)):
    if(i>0 and raw["content_id"][i]==raw["content_id"][i-1]):
        continue
    has=False
    for j in range(len(standard_sub)):
        if (tf_idf.values[i][j] > threshold):
            has=True
            result_tfidf.append([raw["content_id"].values[i],sp_content.values[i][j],standard_sub[j],raw["sentiment_value"].values[i],None,sp_word[i]])
    if(has==False):
        result_tfidf.append([raw["content_id"].values[i],sp_content.values[i][0],standard_sub[0], raw["sentiment_value"].values[i], None,sp_word[i]])
result_tfidf=pd.DataFrame(result_tfidf)
result_tfidf.columns=["content_id","content","subject","sentiment_value","sentiment_word","sp_word"]
result_tfidf["sentiment_value"]=result_tfidf["sentiment_value"].astype(int)
result_tfidf.to_csv("result_tfidf1.csv",encoding="UTF-8",index=False)