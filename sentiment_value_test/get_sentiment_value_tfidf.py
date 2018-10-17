# coding=gbk
import pandas as pd
import torch
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

raw = pd.read_csv("../train.csv")
content_id = raw["content_id"]
content2 = raw["content"]
subject = raw["subject"]
sentiment_value = raw["sentiment_value"]
subject_dic = {"价格": 0, "内饰": 1, "配置": 2, "安全性": 3, "外观": 4, "操控": 5, "油耗": 6, "空间": 7, "舒适性": 8, "动力": 9}
key_words = \
    ['虚位', 'crv', '2w', 'cx', '难受', '太丑', '轴距', '空调', '轴承', '舒适', '水平', '瞬时', '润滑', '打蜡', '喷漆', '机滤', '清零', '压缩机', '尊贵',
     '经销商', '古城', '全系', '主机', '做工', '大改', '性能', '实车', '丰台', '库存', '情况', '转向', '操控性', '不安', '两拳', '多点', '低于', '刹住', '落户',
     '底盘', '便宜', '后排', '摩擦系数', '市区', '棕色', '座椅', '关税', '太高', '上限', '定速', '设计师', '石景山', '价格', '往前', '加油', '个油', '静音',
     '摩擦力', '车内', '看着', '新款', '靠背', '升级', '万公里', '变速箱', '落地', '有没有', '味道', '雷达', '急刹车', '地毯', '不知', '储物', '风噪', '循环',
     '风噪大', '支臂', '塑料', '改改', '价格便宜', '原厂', '动力', '硬伤', '难看', '厚道', '配置', '实惠', '宽敞', '降档', '转速', 'esp', '更换', '4l',
     '外形', 'xt', '里程', '外观设计', '宝马', 'cd', '车况', '空滤', '增加', '黑色', '不烧', '不减', '现金', '轮胎', '平庸', '最低', '实用', '黑车', '尾灯',
     '面板', '社会', '砰砰', '装甲', '中控', '车价', '银色', '不好谈', '检查', '鲨鱼', '原车', '卡钳', '车商', '设计', '综合', '内饰', '空间', '前车', '垃圾',
     '死硬', '操控', '油耗', '好看', '车衣', '二手车', '线条', '欧蓝德', '纠结', '排量', '一家', '超车', '冷气', '缺点', '曼牌', '满意', '方向盘', '刹车油',
     '耐看', '适时', '被动', '漂亮', 'gla', '颠簸', '图像', '滤芯', '机油', '安全性', '畅行', '主打', '均速', '护板', '刹车', '优惠', '指南', '13',
     '刹车片', '歌乐', '太小', '磨损', '现款', '行驶', '浙江', '不好', '精致', '省油', '等待时间', '估计', '发动机', '4s店', '庞大', '车尾', '高德', '不行',
     '安全带', '钢梁', '舒适性', '前脸', '噪音', '异响', '气囊', '菲罗多', 'fb', '事件', '刹车灯', '獠牙', 'nx', '避震', '显示', '表显', '刹车踏板', '室盖',
     'edfc', '裸车', '细节', '召回', '4s', '主动', '空气', '齐亮', '大气', 'gps', '雪地', 'q5', '地力', '保养', '对置', '震动', '视野', '92',
     '配件', '需求', '东西', '不住', '11', '公里', '滤清器', '长城', '价格不菲', '胶套', '文件', '冰雪', '国产', '导航', '踩得', '12', '时速', '出风口',
     '助力', 'xv', '自驾游', '胜出', '傲虎', '加速', '启动', '斯柯达', '铝合金', '颜色', '影响', '太贵', '恭喜', 'model', '用料', '声音', '工作', '2019',
     '平均', '16', '自动', '建议', '系统', '音质', '调节', '发电机', '共振', '提供', '潍坊', '米其林', '故障', '225', '不支', '锈斑', '发现', '大鼻子',
     '性价比', '功能', '10', '外观', '车漆', '冬季', '百公里', '金麒麟', '乘坐', '消耗', '合成', '简陋', '维修', '选择', '山寨', '温度', 'abs', '感应',
     '后备箱', '比比', '摆动', '日常', '材料', '效果', '刹车盘', '套子', '故障率', '车身', '更好', '爆震', '尾翼', '下降', '解决']
temp = pd.read_csv("tf_idf.csv", encoding="gbk")
standard_sub = ['价格-1', '价格0', '价格1', '内饰-1', '内饰0', '内饰1', '配置-1', '配置0', '配置1', '安全性-1', '安全性0', '安全性1', '外观-1',
                '外观0', '外观1', '操控-1', '操控0', '操控1', '油耗-1', '油耗0', '油耗1', '空间-1', '空间0', '空间1', '舒适性-1', '舒适性0', '舒适性1',
                '动力-1', '动力0', '动力1']

words_tfidf = temp[key_words]
values = [[] for i in range(len(raw))]
for i in range(len(raw)):
    for j in range(30):
        v = 0
        words = jieba.lcut(raw["content"].values[i])
        for word in words:
            if (word in words_tfidf.columns.tolist()):
                v += words_tfidf[word].values[j]
        values[i].append(v)
values = pd.DataFrame(values)
values.columns = standard_sub
data = []
for i in range(len(values)):
    sum = 0
    for j in range(len(values.values[i])):
        sum += values.values[i][j]
    for j in range(len(values.values[i])):
        values.values[i][j] /= sum
values.to_csv("values.csv",encoding="UTF-8")


def get_accuracy(threshold=0.32):
    subject_contains = []
    T=0
    i = 0
    while (i < len(content2)):
        index=0
        value=0
        for j in range(len(values.values[i])):
            if(values.values[i][j]>value):
                value=values.values[i][j]
                index=j%3-1
        if value<threshold:
            index=0
        if index==sentiment_value[i]:
            T+=1
        i+=1
    P=T/len(content2)
    return P

#values=pd.read_csv("values.csv",encoding=UTF-8)
cp = values.copy()
for j in range(1, 100):
    values = cp.copy()
    print(get_accuracy(0.01*j), j)
