#coding:utf-8
import configparser as cp
import json

config = cp.RawConfigParser()
config.read('./config/config.cfg')

orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualise = config.getboolean("hog", "visualise")
transform_sqrt = config.getboolean("hog", "transform_sqrt")
block_norm=config.get("hog", "block_norm")

clf_type = config.get("type", "clf_type")

nr_octaves=config.getint("surf", "nr_octaves")
nr_scales=config.getint("surf", "nr_scales")
initial_step_size=config.getint("surf", "initial_step_size")
threshold=config.getfloat("surf", "threshold")
max_points=config.getint("surf", "max_points")
descriptor_only=config.getboolean("surf", "descriptor_only")
n_clusters=config.getint("surf", "n_clusters")

penalty=config.get("LinearSVC","penalty")
loss=config.get("LinearSVC","loss")
dual=config.getboolean("LinearSVC", "dual")
tol=config.getfloat("LinearSVC", "tol")
C=config.getfloat("LinearSVC", "C")
multi_class=config.get("LinearSVC","multi_class")
fit_intercept=config.getboolean("LinearSVC", "fit_intercept")
intercept_scaling=config.getint("LinearSVC","intercept_scaling")
verbose=config.getint("LinearSVC","verbose")
max_iter=config.getint("LinearSVC","max_iter")

homemade_size_x=config.getint("other","homemade_size_x")
homemade_size_y=config.getint("other","homemade_size_y")

homemade_labels={
    0:'剪刀',
    1: '汽车',
    2: '蟒蛇',
    3: '猫',
    4: '男人',
    5: '狗',
    6: '手机',
    7: '乌龟',
    8: '鱼'
    }

cifar_10_labels={
    0:'飞机',
    1: '汽车',
    2: '鸟',
    3: '猫',
    4: '鹿',
    5: '狗',
    6: '青蛙',
    7: '马',
    8: '船',
    9: '货车'
   }
cifar_100_labels={
    0:'水生哺乳动物',
    1: '鱼',
    2: '花卉',
    3: '食品容器',
    4: '水果和蔬菜',
    5: '家用电器',
    6: '家用家具',
    7: '昆虫',
    8: '大型食肉动物',
    9: '大型人造户外用品',
    10:'大自然的户外场景',
    11: '大杂食动物和食草动物',
    12: '中型哺乳动物',
    13: '非昆虫无脊椎动物',
    14: '人',
    15: '爬行动物',
    16: '小型哺乳动物',
    17: '树木',
    18: '车辆1',
    19: '车辆2',
   }
stl_10_labels={
    1:'飞机',
    2: '鸟',
    3: '汽车',
    4: '猫',
    5: '鹿',
    6: '狗',
    7: '马',
    8: '猴',
    9: '船',
    10: '卡车'
    }

