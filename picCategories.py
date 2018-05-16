#coding:utf-8
import sklearn
import numpy as np
import glob
import os
from config import *
import cv2
import skimage
from skimage.feature import hog
from skimage import io
import mahotas
from mahotas.features import surf


def picHogCategories(labels,picPath,model_path,size=(32,32)):
    clf = sklearn.externals.joblib.load(model_path)#加载训练模型
    total=0
    corrnum=0
    for picDir in os.listdir(picPath):
        total+=1
        filename=os.path.join(picPath, picDir)
        #pic =cv2.imdecode(np.fromfile(filename, dtype = np.uint8), cv2.IMREAD_GRAYSCALE)  
        pic=io.imread(filename,as_grey=True)
        pic=skimage.transform.resize(pic, size)
        picFeat = hog(pic,block_norm=block_norm, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=visualise,
         transform_sqrt=transform_sqrt)
        result =int(clf.predict(picFeat.reshape((1, -1)))[0])#预测
        if labels[result] in picDir:
            corrnum+=1
        yield(picDir+'识别为：'+labels[result])
    yield('正确率为：'+str(corrnum*100//total)+'%')

def picHomemadeHogCategories(labels,picPath,model_path):
    for info in picHogCategories(labels,picPath,model_path,(homemade_size_x,homemade_size_y)):
        yield info



def picCifarHogCategories(labels,picPath,model_path):
    for info in picHogCategories(labels,picPath,model_path,(32,32)):
        yield info

def picStlHogCategories(labels,picPath,model_path):
    for info in picHogCategories(labels,picPath,model_path,(96,96)):
        yield info

def picStlSurfCategories(labels,picPath,model_path):
    for info in picSurfCategories(labels,picPath,model_path,(32,32)):
        yield info

def picHomemadeSurfCategories(labels,picPath,model_path):
    for info in picSurfCategories(labels,picPath,model_path,(homemade_size_x,homemade_size_y)):
        yield info

def picSurfCategories(labels,picPath,model_path,size=(32,32)):
    estimator=sklearn.externals.joblib.load(model_path+'.k')
    clf = sklearn.externals.joblib.load(model_path)#加载训练模型

    total=0
    corrnum=0
    for picDir in os.listdir(picPath):
        total+=1
        filename=os.path.join(picPath, picDir)
        #pic =cv2.imdecode(np.fromfile(filename, dtype = np.uint8), cv2.IMREAD_GRAYSCALE)  
        pic=io.imread(filename,as_grey=True)
        pic=skimage.img_as_ubyte(skimage.transform.resize(pic, size))
        picFeat = surf.surf(pic,nr_octaves=nr_octaves,nr_scales=nr_scales,initial_step_size=initial_step_size,threshold=threshold,max_points=max_points,descriptor_only=descriptor_only)
        
        if picFeat.shape!=(0,64):
            clusters = estimator.predict(picFeat)
            features = np.bincount(clusters)#0-n的范围得到n+1个数分别表示0-n每个数出现的次数
            if len(features) < n_clusters:
                features = np.append(features, np.zeros((1, n_clusters-len(features))))

            result =int(clf.predict(features.reshape((1, -1)))[0])#预测
            if labels[result] in picDir:
                corrnum+=1
            yield(picDir+'识别为：'+labels[result])
        else:
            yield(picDir+'该图提取特征失败')
    yield('正确率为：'+str(corrnum*100//total)+'%')

