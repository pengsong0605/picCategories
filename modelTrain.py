#coding:utf-8
import skimage
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.svm import LinearSVC
from skimage import io
import numpy as np
import sklearn
import os
import pickle
import glob
import wx
from config import *
import mahotas
from mahotas.features import surf
import cv2
import gc
from sklearn.cluster import MiniBatchKMeans

def train_saveModel(trainType,picSetType,filePath,model_path):
    yield('开始读取%s数据'%picSetType)
    if picSetType=='cifar-100':
        train_images,train_labels,test_images,test_labels = getCifar100Data(filePath)
    elif picSetType=='cifar-10':
        train_images,train_labels,test_images,test_labels = getCifar10Data(filePath)
    elif picSetType=='stl-10':
        train_images,train_labels,test_images,test_labels =getStl10Data(filePath)
    elif picSetType=='homemade':
        train_images,train_labels,test_images,test_labels=getHomemadeDate(filePath)
    yield('数据读取完成')
    if not test_images.size:
        yield('测试数据为空')
    if not train_images.size:
        yield('训练数据为空')
        return
    if (trainType=='Hog'):
        yield('提取HOG特征')
        for info in getHogFeat(train_images, test_images):
            if type(info)!=tuple:
                yield info
            else:
                trainFds,testFds=info
    elif(trainType=='Surf'):
        if(picSetType.startswith('cifar')):
            yield('该图集不支持surf提取特征')
            return
        yield('提取SURF特征')
        for info in getSurfFeat(model_path,train_images,train_labels,test_images,test_labels):
            if type(info)!=tuple:
                yield info
            else:
                trainFds,train_labels,testFds,test_labels=info


    if (picSetType!='stl-10')and (not trainFds[0].size):
        yield('提取的特征为空，即将结束本次提取')
        return
    #手动释放内存
    del train_images
    del test_images
    gc.collect()
    num = 0
    yield('开始模型训练')
    #Fds都是list(np.array),labels都是array(lenth,)
    if clf_type == 'LinearSVC':
        clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, verbose=verbose,  max_iter=max_iter)
        clf.fit(trainFds, train_labels)
        sklearn.externals.joblib.dump(clf, model_path)#保存训练模型
        yield('保存模型')
        if testFds:
            for i in range(test_labels.size):
                result = clf.predict(testFds[i].reshape((1, -1)))#进行预测    
                if int(result[0]) == int(test_labels[i]):
                    num += 1
            rate = float(num)/len(testFds)
            yield ('测试集测试分类准确率 %f'%rate)
    #手动释放内存
    del trainFds
    del train_labels
    del testFds
    del test_labels
    gc.collect()
#########################################################################
def unpickle(file):
    if os.path.getsize(file) > 0:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    else:
        wx.MessageBox("file not exit",'error')
        return
def getHomemadeDate(filePath):
    try:
        lenthLabel=len(homemade_labels)
        images = [ [] for i in range(lenthLabel)]
        labels = [ [] for i in range(lenthLabel)]
        new_homemade_labels={v:k for k,v in homemade_labels.items()}
        i=0
        for labelStr in os.listdir(filePath):
            labelInt=new_homemade_labels[labelStr]
            dir=os.path.join(filePath,labelStr)
            for picName in os.listdir(dir):
                filename=os.path.join(dir, picName)
                pic=io.imread(filename)         
                pic_tmp=skimage.transform.resize(pic, (homemade_size_x,homemade_size_y,3))
                del pic
                gc.collect()
                images[i].append(pic_tmp)
                labels[i].append(labelInt)
            i+=1
        train_images=[]
        train_labels=[]
        test_images=[]
        test_labels=[]
        for i in range(lenthLabel):
            train_lenth=int(len(images[i])*0.8)
            if train_lenth!=0:
                train_images.append(images[i][:train_lenth])
                train_labels.append(labels[i][:train_lenth])
                test_images.append(images[i][train_lenth:])
                test_labels.append(labels[i][train_lenth:])
            else:
                break
        del images[:]
        del labels[:]
        gc.collect()
        train_images_tmp=np.array(train_images).reshape(-1,homemade_size_x,homemade_size_y,3)
        train_labels_tmp=np.array(train_labels).reshape(1,-1)[0]
        test_images_tmp=np.array(test_images).reshape(-1,homemade_size_x,homemade_size_y,3)
        test_labels_tmp=np.array(test_labels).reshape(1,-1)[0]
        del train_images
        del train_labels
        del test_images
        del test_labels
        gc.collect()
        return train_images_tmp,train_labels_tmp,test_images_tmp,test_labels_tmp
    except MemoryError as e:  
        #('爆内存了,图片太大')
        t=np.array([])
        return t,t,t,t


def getStl10Data(filePath):

    for childName in os.listdir(filePath):
        fd = os.path.join(filePath, childName)
        if os.path.getsize(fd) > 0:
            if childName=='test_X.bin':
                with open(fd, 'rb') as f:
                    everything1 = np.fromfile(f, dtype=np.uint8)
                    test_image = np.reshape(everything1, (-1, 3, 96, 96))
                    test_images = np.transpose(test_image, (0, 3, 2, 1))
        
            if childName=='test_y.bin':
                with open(fd, 'rb') as f:
                    test_label = np.fromfile(f, dtype=np.uint8)
                    test_labels=np.reshape(test_label, (1, -1))
                    test_labels=test_labels.astype(np.int)


            if childName=='train_X.bin':
                with open(fd, 'rb') as f:
                    everything2 = np.fromfile(f, dtype=np.uint8)
                    train_image = np.reshape(everything2, (-1, 3, 96, 96))
                    train_images = np.transpose(train_image, (0, 3, 2, 1))

            if childName=='train_y.bin':
                with open(fd, 'rb') as f:
                    train_label = np.fromfile(f, dtype=np.uint8)
                    train_labels=np.reshape(train_label, (1, -1))
                    train_labels=train_labels.astype(np.int)
    #np.array(5000, 96, 96, 3),np.array(5000,)np.array(8000, 96, 96, 3),np.array(8000,),
    return train_images,train_labels[0],test_images,test_labels[0]


def getCifar10Data(filePath):
    train_images = []
    train_labels = []
    for childName in os.listdir(filePath):
        f = os.path.join(filePath, childName)
        if childName.startswith('data_batch_'):       
            data = unpickle(f)
            train = np.reshape(data[b'data'], (-1, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (1, -1))
            train_images.extend(train)
            train_labels+=labels.tolist()[0]
        if childName=='test_batch':
            data = unpickle(f)
            test_images = np.reshape(data[b'data'], (-1, 3, 32 * 32))
            test_labels = np.reshape(data[b'labels'], (1, -1))   
    train_images=np.array(train_images).reshape(-1,3,32,32).transpose(0,2,3,1)
    test_images=test_images.reshape(-1,3,32,32).transpose(0,2,3,1)
    #np.array(50000, 32, 32, 3),np.array(50000,),np.array(10000, 32, 32, 3),np.array(10000,)
    return train_images,np.array(train_labels), test_images,test_labels[0]
    

def getCifar100Data(filePath):
    for childName in os.listdir(filePath):
        f = os.path.join(filePath, childName)
        if childName == 'train':
            data = unpickle(f)
            train_images = np.reshape(data[b'data'], (-1, 3, 32 * 32))
            train_labels = np.reshape(data[b'coarse_labels'], (1, -1))
        if childName=='test':         
            data = unpickle(f)
            test_images = np.reshape(data[b'data'], (-1, 3, 32 * 32))
            test_labels = np.reshape(data[b'coarse_labels'], (1, -1))
    train_images=train_images.reshape(-1,3,32,32).transpose(0,2,3,1)
    test_images=test_images.reshape(-1,3,32,32).transpose(0,2,3,1)
    #np.array(50000, 32, 32, 3),np.array(50000,),np.array(10000, 32, 32, 3),np.array(10000,)
    return train_images,train_labels[0], test_images,test_labels[0]

def getHogFeat(train_images, test_images):
    yield('开始提取HOG特征')
    testFds=[]
    trainFds=[]
    if test_images.size:
        for data in test_images:
            
            gray = rgb2gray(data)
            fd = hog(gray,block_norm=block_norm, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,visualise=visualise, transform_sqrt=transform_sqrt)
            testFds.append(fd)
        yield('提取完测试特征')
    for data in train_images:
        gray = rgb2gray(data)
        fd = hog(gray,block_norm=block_norm, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,visualise=visualise, transform_sqrt=transform_sqrt)
        trainFds.append(fd)
    yield('提取完训练特征')
    yield trainFds,testFds

def getSurfFeat(model_path,train_images,train_labels,test_images,test_labels):
    yield('开始提取SURF特征')
    testFds=[]
    trainFds=[]
    if test_images.size:
        for data in test_images:
            gray = skimage.img_as_ubyte(rgb2gray(data))
            fd = surf.surf(gray,nr_octaves=nr_octaves,nr_scales=nr_scales,initial_step_size=initial_step_size,threshold=threshold,max_points=max_points,descriptor_only=descriptor_only)
            testFds.append(fd)
        yield('提取完测试特征')
    for data in train_images:
        gray = skimage.img_as_ubyte(rgb2gray(data))
        fd = surf.surf(gray,nr_octaves=nr_octaves,nr_scales=nr_scales,initial_step_size=initial_step_size,threshold=threshold,max_points=max_points,descriptor_only=descriptor_only)
        trainFds.append(fd)
    yield('提取完训练特征')

    
    #训练集聚类
    yield('特征聚类')
    train_surf_features = np.concatenate(trainFds)
    #print(train_surf_features.shape)

    estimator = MiniBatchKMeans(n_clusters=n_clusters)
    estimator.fit_transform(train_surf_features)#fit_transform()先拟合数据，再标准化 
    del train_surf_features
    gc.collect()

    yield('正在生成训练词袋')
    #用kmeans的数据来预测训练集，生成词袋大小为301，词袋就是该图surf特征的别名
    X_train = []
    i=0
    for instance in trainFds:
        if instance.shape!=(0,64):
            clusters = estimator.predict(instance)
            features = np.bincount(clusters)#0-n的范围得到n+1个数分别表示0-n每个数出现的次数
            if len(features) < n_clusters:
                features = np.append(features, np.zeros((1, n_clusters-len(features))))
            X_train.append(features)
        else:
            train_labels_copy=np.delete(train_labels,i,axis=0)
            del train_labels
            gc.collect()
            train_labels=train_labels_copy
            i-=1
        i+=1

    del trainFds
    gc.collect()

    yield('正在生成测试词袋')
    #得到测试集surf特征的词袋
    X_test = []
    i=0
    for instance in testFds:
        if instance.shape!=(0,70):
            clusters = estimator.predict(instance)
            features = np.bincount(clusters)
            if len(features) < n_clusters:
                features = np.append(features, np.zeros((1, n_clusters-len(features))))
            X_test.append(features)
        else:
            test_labels_copy=np.delete(test_labels,i,axis=0)
            del test_labels
            gc.collect()
            test_labels=test_labels_copy
            i-=1
        i+=1

    del testFds
    gc.collect()

    sklearn.externals.joblib.dump(estimator, model_path+'.k')

    yield X_train,train_labels,X_test,test_labels 





    





