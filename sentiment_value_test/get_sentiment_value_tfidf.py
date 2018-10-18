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
    ['前车', '平庸', '肯定', '庞大', '虚位', '舒适', '大改', '米其林', '主机', '外观设计', '2019', '车内', '不好', '摆动', '獠牙', '价格', '地力', '够用',
     '消耗', '底盘', '冬季', '傲虎', '漂亮', '越野', '检查', '操控性', '四驱', '外形', '前脸', '行驶', '冬天', '冷气', '刹住', '实惠', '真皮', '不行', '瞬时',
     '机油', '进口', '刹车', '社会', '安卓', '日常', '车身', '胜出', '舒适性', '建议', '两拳', '百公里', '投诉', '好看', '声音', '车衣', '确实', '后备箱',
     '更好', '缺点', '尾灯', '隔音', '耐看', '共振', '鲨鱼', '导航', '刹车踏板', '储物', '4l', '加热', '平均', '车商', '座椅', '能力', '简陋', '个油', '对置',
     '增加', '手机', '全系', '合成', '异响', '刹车片', '斯柯达', '故障', '踩得', '现款', '好用', '套子', '性能', '视野', '优惠', 'cd', '摩擦系数', '倒车',
     '方向盘', '新指', '上限', '自动', '硬伤', '高速', '效果', '主打', '太丑', '顿挫', '途观', '召回', '维修', '空调', '铝合金', '市区', '动力', '14', '颜色',
     'abs', '锈斑', '白色', 'q5', '公里', '起步', '万公里', '发现', '做工', '30', '乘坐', 'nx', '自驾游', '汉兰达', '循环', '看着', '变速箱', '爆震',
     '配置', '太高', '全时', '轴承', '鹰眼', '故障率', '性价比', '开关', '新款', '路况', '一家', '排量', '恭喜', '改改', '缺陷', '低于', '一点', '潍坊', '齐亮',
     '不安', '震动', '打蜡', '尊贵', '太小', '发动机', '轿车', '精致', '砰砰', '不减', '冰雪', '实车', '指南', '音质', '时尚', '不支', '油耗', '加速', '降低',
     '宝马', '长城', '事件', '室盖', '石景山', '森林', '纠结', '解决', '比比', '急刹车', '主动', '难受', '空间', '宽敞', '大气', '斯巴鲁', '护板', '静音',
     'suv', '92', '费油', '味道', 'cvt', '音响', '安全性', '火花塞', '提供', '奇骏', '卡钳', '支臂', '操控', '系统', '气囊', '表显', '11', '舒服',
     '浙江', '润滑', '钢梁', '歌乐', '欧蓝德', '提升', '油门', 'model', '制冷', '高德', '超车', '配件', '方向', '丰台', '山寨', '价格不菲', '被动', '不错',
     'xv', '车漆', '国产', '大鼻子', '用料', '18', '省油', '线条', '噪音', '等待时间', '设计', '10', '价格低', '喜欢', '外观', '颠簸', 'crv', '实用',
     '选择', '最低', '能比', '下降', '综合', '保养', '太贵', '4s店', '垃圾', '后排', '平台', '启动', '轮胎', 'gla', 'esp', '材料', '不值', '感应',
     '雪地', '不好谈', '内饰', '降档', '设计师', '地图', '豪华', '水平', '汽油', '不住', '畅行', '出风口', '4s', '死硬', '古城', '往前', '刹车盘', '落户',
     '风噪大', '中控', '不贵', '便宜', '价格便宜', '事实', '驾驶', '95', '蔚揽', '摩擦力', 'edfc', '厚道', '难看', '黑车', '满意', '滤清器', '感觉', '塑料']

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
