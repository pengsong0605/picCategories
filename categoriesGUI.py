#!python3
#coding=utf-8
import wx
import os
import cv2
import time
import _thread
import numpy  
from modelTrain import train_saveModel
from picCategories import *


class categoriesFrame(wx.Frame):
    #用于显示程序提示信息
    promptMessageText=''
    #初始化
    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(categoriesFrame, self).__init__(*args, **kw)
        self.makePanel()
        self.makeMenu()
       # self.makeTool()
        self.makeIco()
        self.Centre()
        self.count=True
        self.count1=True


        
    # 初始化面板
    def makePanel(self):
        #创建两个面板p1、p2，p1用于训练和保存模型，p2用于创建子面板
        sp=wx.SplitterWindow(self,style=wx.SP_LIVE_UPDATE)# 创建一个分割窗,parent是frame
        p1=wx.Panel(sp,style=wx.SUNKEN_BORDER)  #创建子面板p1
        p2=wx.Panel(sp,style=wx.SUNKEN_BORDER)  # 创建子面板p2
        sp.SplitHorizontally(p1, p2, 150)#竖直分割面板
        box1 = wx.BoxSizer(wx.VERTICAL)#创建一个垂直布局
        
        box1_1=wx.BoxSizer(wx.HORIZONTAL)#创建一个水平布局
        picSetType = wx.StaticText(p1,style = wx.ALIGN_LEFT) 
        picSetType.SetLabel('选择图片集类型:') 
        box1_1.Add(picSetType, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        picTypes = ['cifar-10', 'cifar-100', 'stl-10','homemade'] 
        self.picSetTypeChoice = wx.Choice(p1,style = wx.ALIGN_CENTER,choices = picTypes)
        box1_1.Add(self.picSetTypeChoice, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)

        trainType = wx.StaticText(p1,style = wx.ALIGN_LEFT) 
        trainType.SetLabel('选择训练类型:') 
        box1_1.Add(trainType, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        trainTypes = ['Hog', 'Surf'] 
        self.trainTypeChoice = wx.Choice(p1,style = wx.ALIGN_CENTER,choices = trainTypes)
        box1_1.Add(self.trainTypeChoice, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)

        box1.Add(box1_1)

        box1_2=wx.BoxSizer(wx.HORIZONTAL)#创建一个水平布局
        picSetDir = wx.StaticText(p1,style = wx.ALIGN_LEFT) 
        picSetDir.SetLabel('图片集地址:      ') 
        box1_2.Add(picSetDir, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        self.picSetDirTextCtrl = wx.TextCtrl(p1,style = wx.TE_READONLY|wx.TE_LEFT,size=(250,25)) 
        box1_2.Add(self.picSetDirTextCtrl,0,wx.ALIGN_LEFT|wx.ALL,5)
        picSetDirButton=wx.Button(p1, label='1.打开', size=(50,25))
        picSetDirButton.Bind(wx.EVT_BUTTON,self.getDir)
        box1_2.Add(picSetDirButton, 1, wx.ALL,5)
        trainButton=wx.Button(p1, label='开始训练', size=(50,50))
        trainButton.Bind(wx.EVT_BUTTON,self.startTrain)
        box1_2.Add(trainButton, 1, wx.ALL,5)
        box1.Add(box1_2)

        box1_3=wx.BoxSizer(wx.HORIZONTAL)#创建一个水平布局
        modelDir = wx.StaticText(p1,style = wx.ALIGN_LEFT) 
        modelDir.SetLabel('模型保存地址:   ') 
        box1_3.Add(modelDir, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        self.modelDirTextCtrl = wx.TextCtrl(p1,style = wx.TE_READONLY|wx.TE_LEFT,size=(250,25)) 
        box1_3.Add(self.modelDirTextCtrl,0,wx.ALIGN_LEFT|wx.ALL,5)
        modelDirButton=wx.Button(p1, label='2.打开', size=(50,25))
        modelDirButton.Bind(wx.EVT_BUTTON,self.getDir)
        box1_3.Add(modelDirButton, 1, wx.ALL,5)
        box1.Add(box1_3)
        p1.SetSizer(box1)

        #创建两个面板p2_1、p2_2，p2_1用于显示过程，p2_2用于图片分类
        sp2=wx.SplitterWindow(p2,style=wx.SP_LIVE_UPDATE)# 创建一个分割窗,parent是p2
        p2_1=wx.Panel(sp2,style=wx.SUNKEN_BORDER,size=(500,150))  #创建子面板p2_1
        p2_2=wx.Panel(sp2,style=wx.SUNKEN_BORDER)  # 创建子面板p2_2
        sp2.SplitHorizontally(p2_1,p2_2, 150)#竖直分割面板
        box2 = wx.BoxSizer(wx.VERTICAL)#创建一个垂直布局
        box2.Add(sp2, 1, wx.EXPAND)#将子分割窗布局延伸至整个p2空间
        p2.SetSizer(box2)

        box2_1 = wx.BoxSizer(wx.VERTICAL)#创建一个垂直布局
        self.prompt_message = wx.TextCtrl(p2_1,style = wx.TE_LEFT|wx.TE_MULTILINE|wx.TE_READONLY | wx.BORDER_NONE|wx.TE_RICH2 ) 
        box2_1.Add(self.prompt_message,1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        p2_1.SetSizer(box2_1)
        self.showInfo('欢迎使用本程序，上面是训练模块，下面是分类模块')
        
        box2_2 = wx.BoxSizer(wx.VERTICAL)#创建一个垂直布局

        box2_2_3=wx.BoxSizer(wx.HORIZONTAL)#创建一个水平布局
        modelType = wx.StaticText(p2_2,style = wx.ALIGN_LEFT) 
        modelType.SetLabel('选择模型类型:  ') 
        box2_2_3.Add(modelType, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        self.modelTypeChoice = wx.Choice(p2_2,style = wx.ALIGN_CENTER,choices = picTypes)
        box2_2_3.Add(self.modelTypeChoice, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)

        trainsType = wx.StaticText(p2_2,style = wx.ALIGN_LEFT) 
        trainsType.SetLabel('选择训练类型:') 
        box2_2_3.Add(trainsType, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        self.trainTypesChoice = wx.Choice(p2_2,style = wx.ALIGN_CENTER,choices = trainTypes)
        box2_2_3.Add(self.trainTypesChoice, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)

        box2_2.Add(box2_2_3)

        box2_2_1=wx.BoxSizer(wx.HORIZONTAL)#创建一个水平布局
        modelSavaDir = wx.StaticText(p2_2,style = wx.ALIGN_LEFT) 
        modelSavaDir.SetLabel('模型地址:        ') 
        box2_2_1.Add(modelSavaDir, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        self.modelSavaDirTextCtrl = wx.TextCtrl(p2_2,style = wx.TE_READONLY|wx.TE_LEFT,size=(250,25)) 
        box2_2_1.Add(self.modelSavaDirTextCtrl,0,wx.ALIGN_LEFT|wx.ALL,5)
        modelSavaDirButton=wx.Button(p2_2, label='3.打开', size=(50,25))
        modelSavaDirButton.Bind(wx.EVT_BUTTON,self.getDir)
        box2_2_1.Add(modelSavaDirButton, 1, wx.ALL,5)
        categoriesButton=wx.Button(p2_2, label='开始分类', size=(50,50))
        categoriesButton.Bind(wx.EVT_BUTTON,self.startCategories)
        box2_2_1.Add(categoriesButton, 1, wx.ALL,5)
        box2_2.Add(box2_2_1)

        box2_2_2=wx.BoxSizer(wx.HORIZONTAL)#创建一个水平布局
        picsDir = wx.StaticText(p2_2,style = wx.ALIGN_LEFT) 
        picsDir.SetLabel('分类图片地址:  ') 
        box2_2_2.Add(picsDir, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        self.picsDirTextCtrl = wx.TextCtrl(p2_2,style = wx.TE_READONLY|wx.TE_LEFT,size=(250,25)) 
        box2_2_2.Add(self.picsDirTextCtrl,0,wx.ALIGN_LEFT|wx.ALL,5)
        picsDirButton=wx.Button(p2_2, label='4.打开', size=(50,25))
        picsDirButton.Bind(wx.EVT_BUTTON,self.getDir)
        box2_2_2.Add(picsDirButton, 1, wx.ALL,5)
        box2_2.Add(box2_2_2)
        p2_2.SetSizer(box2_2)

        
        self.Centre() 
        self.Show() 
        self.Fit()  

    #加载图标
    def makeIco(self):
        icon = wx.Icon()
        icon.LoadFile(r"ico\48.ico", wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

    #菜单栏(100-120)
    def makeMenu(self):
        menubar = wx.MenuBar()

        #菜单选项-
        exitMenu=wx.Menu()
        aboutMenu=wx.Menu()


        #文件菜单
        exitItem=wx.MenuItem(exitMenu,id=100,text="退出\tCtrl+X",helpString="退出该软件")
        self.Bind(wx.EVT_MENU,self.exitDeal,id=100)
        exitMenu.Append(exitItem)

        #关于
        aboutItem = wx.MenuItem(aboutMenu,id=101,text="关于软件\tCtrl+A")
        self.Bind(wx.EVT_MENU,self.aboutDeal,id=101)
        aboutMenu.Append(aboutItem)   

        #绑定到菜单栏
        menubar.Append(exitMenu, '&退出')
        menubar.Append(aboutMenu, '&关于')
        self.SetMenuBar(menubar)


    def startCategories(self,event):
        #训练集不同标签不同
        def inlineF():
            try:
                trainTypes=self.trainTypesChoice.GetString(self.trainTypesChoice.GetSelection())
                modelType=self.modelTypeChoice.GetString(self.modelTypeChoice.GetSelection())
                picPath=self.picsDirTextCtrl.GetLabelText()
                modelPath=self.modelSavaDirTextCtrl.GetLabelText()
                if trainTypes!='' and modelType!='' and picPath!= ''and modelPath!='':
                    if(self.count1):
                        self.count1=False

                        if modelType=='cifar-10':
                            if trainTypes=='Hog':
                                for info in picCifarHogCategories(cifar_10_labels,picPath,modelPath):
                                    self.showInfo(info)
                            else:
                                self.showInfo('不支持该操作')
                        elif modelType=='cifar-100':
                            if trainTypes=='Hog':
                                for info in picCifarHogCategories(cifar_100_labels,picPath,modelPath):
                                    self.showInfo(info)
                            else:
                                self.showInfo('不支持该操作')
                        elif modelType=='stl-10':
                            if trainTypes=='Hog':
                                for info in picStlHogCategories(stl_10_labels,picPath,modelPath):
                                    self.showInfo(info)
                            elif trainTypes=='Surf':
                                for info in picStlSurfCategories(stl_10_labels,picPath,modelPath):
                                    self.showInfo(info)
                        elif modelType=='homemade':
                            if trainTypes=='Hog':
                                for info in picHomemadeHogCategories(homemade_labels,picPath,modelPath):
                                    self.showInfo(info)
                            elif trainTypes=='Surf':
                                for info in picHomemadeSurfCategories(homemade_labels,picPath,modelPath):
                                    self.showInfo(info)
                                
                        self.count1=True
                    else:
                        wx.MessageBox('请等待此次分类结束','提示')
                else:
                    wx.MessageBox('请选则图片集地址、模型保存地址','提示')
            except Exception as e:  
                wx.MessageBox('%s'%e,'error')
                self.count1=True
        _thread.start_new_thread(inlineF,())



    def startTrain(self,event):
        #训练集不同类型数量不同
        def inlineF():
            try:
                trainType=self.trainTypeChoice.GetString(self.trainTypeChoice.GetSelection())
                picSetType=self.picSetTypeChoice.GetString(self.picSetTypeChoice.GetSelection())
                file_path=self.picSetDirTextCtrl.GetLabelText()
                model_path=self.modelDirTextCtrl.GetLabelText()
                if trainType!='' and picSetType!='' and file_path!='' and model_path!='':
                    if(self.count):
                        self.count=False
                        for info in train_saveModel(trainType,picSetType,file_path,model_path):
                            self.showInfo(info)
                        self.count=True
                    else:
                        wx.MessageBox('请等待此次训练结束','提示')
                else:
                    wx.MessageBox('请选择类型、图片集地址、模型保存地址','提示')
            except MemoryError as e:
                wx.MessageBox('爆内存了，多半是训练炸了')
                self.count=True
            except Exception as e:  
                wx.MessageBox('%s'%e,'error')
                self.count=True
        _thread.start_new_thread(inlineF,())


    def getDir(self,event):
        try:     
            tmp=event.GetEventObject().GetLabel()
            if(tmp=='1.打开'):
                dialog = wx.DirDialog(self,"选择路径",os.getcwd(),style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
                if dialog.ShowModal() == wx.ID_OK:
                    dir = dialog.GetPath()
                    self.picSetDirTextCtrl.SetLabel(dir)
            elif (tmp=='2.打开'):
                dialog = wx.FileDialog(self, message ="保存文件", wildcard = "All files (*.*)|*.*", style = wx.FD_SAVE)
                if dialog.ShowModal() == wx.ID_OK:
                    dir=dialog.GetPath()
                    self.modelDirTextCtrl.SetLabel(dir)
            elif (tmp=='3.打开'):
                dialog = wx.FileDialog(self, message ="选择单个文件", wildcard = "All files (*.*)|*.*", style = wx.FD_OPEN)
                if dialog.ShowModal() == wx.ID_OK:
                    dir=dialog.GetPath()
                    self.modelSavaDirTextCtrl.SetLabel(dir)
            elif(tmp=='4.打开'):
                dialog = wx.DirDialog(self,"选择路径",os.getcwd(),style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
                if dialog.ShowModal() == wx.ID_OK:
                    dir = dialog.GetPath()
                    self.picsDirTextCtrl.SetLabel(dir)
        except Exception as e:  
            wx.MessageBox("something is wrong\n%s" %e)  
        #销毁对话框,释放资源.
        dialog.Destroy()
        

    def exitDeal(self,e):
        exit(0)

    def aboutDeal(self,event):
        wx.MessageBox(
            "该软件主要用于图片分类。\n该软件仅用于学术讨论，版权归松裘所有。",
            caption="关于"
            )
    
    def showInfo(self,info):
        self.promptMessageText=(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+' '+info+'\n')+self.promptMessageText
        self.prompt_message.SetLabelText(self.promptMessageText)      
    






if __name__ == '__main__':
    # When this module is run (not imported) then create the app, the
    # frame, show it, and start the event loop.
    app = wx.App()
    frm = categoriesFrame(None, title='categoriesTool',style=wx.DEFAULT_FRAME_STYLE^ (wx.RESIZE_BORDER | wx.MAXIMIZE_BOX ),size=(600,550))
    frm.Show()
    app.MainLoop()


