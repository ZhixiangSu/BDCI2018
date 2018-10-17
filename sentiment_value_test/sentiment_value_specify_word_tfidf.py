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
key_words_raw = \
    ['虚位', '真皮', 'crv', '火花塞', '2w', 'cx', '难受', '太丑', '轴距', '空调', '轴承', '舒适', '水平', '瞬时', '润滑', '打蜡', '喷漆', '机滤', '清零',
     '压缩机', '尊贵', '经销商', '古城', '全系', '主机', '做工', '进口', '轿车', 'suv', '大改', '性能', '路况', '实车', '丰台', '感觉', '库存', '情况',
     '转向', '操控性', '不安', '两拳', '多点', '低于', '刹住', '落户', '底盘', '便宜', '后排', '摩擦系数', '市区', '棕色', '座椅', '关税', '95', '太高',
     '上限', '定速', '设计师', '石景山', '价格', '往前', '加油', '个油', '静音', '30', '影像', '新指', '摩擦力', '车内', '看着', '新款', '靠背', '升级',
     '万公里', '变速箱', '落地', '有没有', '斯巴鲁', '味道', '肯定', '雷达', '急刹车', '地毯', '不知', '储物', '风噪', '一点', '循环', '风噪大', '支臂', '高速',
     '塑料', '改改', '汽油', '价格便宜', '原厂', '动力', '能比', '硬伤', '难看', '厚道', '配置', '实惠', '宽敞', '降档', '转速', 'esp', '更换', '4l',
     '外形', 'xt', '里程', '外观设计', '宝马', 'cd', '车况', '空滤', '增加', '黑色', '不烧', '不减', '现金', '轮胎', '轮毂', '平庸', '最低', '手机',
     'cvt', '实用', '黑车', '尾灯', '缺陷', '面板', '社会', '砰砰', '装甲', '中控', '车价', '银色', '不好谈', '检查', '鲨鱼', '立德', '原车', '卡钳', '车商',
     '设计', '综合', '内饰', '空间', '前车', '垃圾', '死硬', '操控', '油耗', '好看', '车衣', '二手车', '线条', '蓝牙', '欧蓝德', '纠结', '排量', '一家', '超车',
     '冷气', '私信', '缺点', '曼牌', '满意', '方向盘', '蔚揽', '不贵', '刹车油', '耐看', '适时', '被动', '漂亮', 'gla', '颠簸', '图像', '滤芯', '机油',
     '安全性', '汉兰达', '倒车', '畅行', '主打', '习惯', '油门', '均速', '不错', '护板', '刹车', '优惠', '不值', '18', '指南', '价格低', '13', '刹车片',
     '方向', '全时', '歌乐', '开关', '太小', '驾驶', '磨损', '现款', '行驶', '浙江', '不好', '精致', '省油', '等待时间', '估计', '发动机', '4s店', '庞大',
     '车尾', '高德', '不行', '确实', '安全带', '钢梁', '舒适性', '前脸', '噪音', '异响', '气囊', '菲罗多', '楼主', 'fb', '事件', '刹车灯', '獠牙', 'nx',
     '投诉', '避震', '显示', '音响', '奇骏', '能力', '表显', '刹车踏板', '室盖', 'edfc', '裸车', '细节', '提升', '召回', '4s', '主动', '冬天', '顿挫',
     '空气', '白色', '齐亮', '加热', '大气', 'gps', '隔音', '雪地', 'q5', '地力', '保养', '对置', '震动', '视野', '安卓', '92', '配件', '四驱', '120',
     '需求', '东西', '不住', '11', '地图', '公里', '滤清器', '长城', '价格不菲', '胶套', '文件', '喜欢', '冰雪', '14', '国产', '越野', '导航', '踩得',
     '好用', '12', '时速', '出风口', '平台', '助力', 'xv', '自驾游', '胜出', '傲虎', '加速', '启动', '斯柯达', '车载', '铝合金', '颜色', '影响', '太贵',
     '恭喜', 'model', '用料', '声音', '工作', '2019', '平均', '16', '自动', '建议', '系统', '音质', '鹰眼', '调节', '费油', '发电机', '共振', '提供',
     '潍坊', '米其林', '故障', '225', '起步', '不支', '锈斑', '发现', '大鼻子', '手刹', '性价比', '功能', '10', '外观', '车漆', '冬季', '百公里', '金麒麟',
     '乘坐', '消耗', '豪华', '合成', '简陋', '维修', '够用', '选择', '森林', '老款', '山寨', '温度', '事实', 'abs', '感应', '后备箱', '比比', '区别', '途观',
     '摆动', '日常', '舒服', '材料', '效果', '刹车盘', '套子', '故障率', '车身', '更好', '爆震', '降低', '时尚', '尾翼', '下降', '制冷', '解决']
temp = pd.read_csv("tf_idf.csv", encoding="gbk")
standard_sub = ['价格-1', '价格0', '价格1', '内饰-1', '内饰0', '内饰1', '配置-1', '配置0', '配置1', '安全性-1', '安全性0', '安全性1', '外观-1',
                '外观0', '外观1', '操控-1', '操控0', '操控1', '油耗-1', '油耗0', '油耗1', '空间-1', '空间0', '空间1', '舒适性-1', '舒适性0', '舒适性1',
                '动力-1', '动力0', '动力1']
def get_accuracy(threshold=0.09):
    subject_contains = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    i = 0
    while (i < len(content2)):
        k = 0
        while (i + k < len(content2) and content_id[i + k] == content_id[i]):
            if (values.values[i][subject_dic[subject.values[i + k]] * 3 + sentiment_value[i + k] + 1] > threshold):
                TP += 1
            else:
                FN += 1
            values.values[i][subject_dic[subject.values[i + k]] * 3 + sentiment_value[i + k] + 1] = 0
            k += 1
        for j in range(len(standard_sub)):
            if (values.values[i][j] > threshold):
                FP += 1
        i += k
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    return F1, FN, FP

removed = [0 for i in range(len(key_words_raw))]
li=0.445829372205204
try:
    for l in range(len(key_words_raw)):
        key_words=key_words_raw.copy()
        key_words.remove(key_words[l])
        words_tfidf = temp[key_words]
        print(key_words[l])
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
        for i in range(len(values)):
            sum = 0
            for j in range(len(values.values[i])):
                sum += values.values[i][j]
            for j in range(len(values.values[i])):
                values.values[i][j] /= sum
        temp3=get_accuracy()[0]
        print(temp3)
        if(temp3>li):
            print("REMOVED")
            removed[l]=1
except Exception  as e:
    print("error")
final_key=[]
for l in range(len(key_words_raw)):
    if(removed[l]==0):
        final_key.append(key_words_raw[l])
print(final_key)
final_key=pd.DataFrame(final_key)
final_key.to_csv("final_key.csv")
