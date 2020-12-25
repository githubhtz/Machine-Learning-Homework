##Author:Zhu Haotian 2020317110039
##Date:2020/12/21
#coding=utf-8

import os
import numpy as np#进行数组运算
import cv2#跨平台计算机视觉库
import tkinter#脚本图形界面接口
import tkinter.filedialog
from PIL import Image, ImageTk#图像处理标准库

#设置一个100*100的图片大小，用于后续图片的缩放
jpg_size =(100,100)

def Database(path):
    #使用os中的listdir列出路径下的所有文件
    xunlian_files = os.listdir(path)
    #对文件夹下的图片数量计数
    xunlian_number = len(xunlian_files)
    #创建一个空列表
    T = []
    for i in range(1,xunlian_number):
        #使用cv2的imread来读取图片，并将图片以灰度模式加载
        image = cv2.imread(path+'/'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
        #使用cv2的resize来对图片大小进行更改
        image=cv2.resize(image,jpg_size)
        # 把所有图片的像素平铺
        image = image.reshape(image.size,1)
        #使用append来用空列表构建样本集合
        T.append(image)        
    #创建数组
    T = np.array(T)
    #将T转化为T.shape[0]行，T.shape[1]列的矩阵
    T = T.reshape(T.shape[0],T.shape[1])
    return np.mat(T).T
 
def eigenfaceCore(T):
    #对各个列求均值
    m = T.mean(axis = 1)
    #进行0均值化
    A = T-m
    #矩阵A*A的转置定义为L
    L = (A.T)*(A)

    #计算L的特征向量和特征值：V是特征值，D是特征向量
    V, D = np.linalg.eig(L)
    L_eig = []
    for i in range(A.shape[1]):
            L_eig.append(D[:,i])
    L_eig = np.mat(np.reshape(np.array(L_eig),(-1,len(L_eig))))
    #计算L的特征向量
    eigenface = A * L_eig
    return eigenface,m,A  
 
def recognize(testImage, eigenface,m,A):
    _,trainNumber = np.shape(eigenface)
    #投影到特征脸后的
    projectedImage = eigenface.T*(A)
    #对中文路径的支持
    testImageArray = cv2.imdecode(np.fromfile(testImage,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    #使用cv2的resize来对图片大小进行更改
    testImageArray=cv2.resize(testImageArray,jpg_size)
    #将图片像素平铺
    testImageArray = testImageArray.reshape(testImageArray.size,1)
    testImageArray = np.mat(np.array(testImageArray))
    differenceTestImage = testImageArray - m
    projectedTestImage = eigenface.T*(differenceTestImage)
    #测试图片与训练图片计算距离
    distance = []
    for i in range(0, trainNumber):
        q = projectedImage[:,i]
        temp = np.linalg.norm(projectedTestImage - q)
        distance.append(temp)
    #距离最近的两个特征脸视为同一个人的脸
    minDistance = min(distance)
    index = distance.index(minDistance)
    #展示识别到的图像
    cv2.imshow("recognize result",cv2.imread('./TrainDatabase'+'/'+str(index+1 )+'.jpg',cv2.IMREAD_GRAYSCALE))
    cv2.waitKey()
    return index+1
# PCA人脸识别的主程序
def main_pro(filename):
    T = Database('./TrainDatabase')
    eigenface,m,A = eigenfaceCore(T)
    testimage = filename
    print(testimage)
    print(recognize(testimage, eigenface,m,A))


# 使用Tkinter来创建图形界面
def picture():
    root = tkinter.Tk()
    #设置窗口标题
    root.title("PCA人脸识别")
    #点击选择图片时调用
    def select():
        #选择打开的文件，返回文件名
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            #jpg图片文件名和路径
            a=filename
            im=Image.open(a)
            tkimg=ImageTk.PhotoImage(im)
            l.config(image=tkimg)
            btn1.config(command=lambda : main_pro(filename))
            btn1.config(text = "运行")
            btn1.pack()
            root.mainloop()
    #显示图片的位置
    l = tkinter.Label(root)
    l.pack()
    
    btn = tkinter.Button(root,text="请选择需要识别的图片",command=select)
    btn.pack()
    
    btn1 = tkinter.Button(root)
    root.mainloop()

#显示图形界面
picture()
