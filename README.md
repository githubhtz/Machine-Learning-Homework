# Machine-Learning-Homework
Machine Learning course assignments
# Prerequisites
python 2.7.2
numpy 1.19.4
matplotlib 3.3.3
cv2 4.4.0
PIL 8.0.1
# 数据集介绍
ORL人脸数据集共包含40个不同人的400张图像，是在1992年4月至1994年4月期间由英国剑桥的Olivetti研究实验室创建。
此数据集下包含40个目录，每个目录下有10张图像，每个目录表示一个不同的人。所有的图像是以PGM格式存储，灰度图，图像大小宽度为92，高度为112。对每一个目录下的图像，这些图像是在不同的时间、不同的光照、不同的面部表情(睁眼/闭眼，微笑/不微笑)和面部细节(戴眼镜/不戴眼镜)环境下采集的。所有的图像是在较暗的均匀背景下拍摄的，拍摄的是正脸(有些带有略微的侧偏)。
# 项目流程图
![iamge1](https://github.com/githubhtz/image/blob/main/1.png)
# 项目代码操作步骤
使用python运行程序，弹出窗口，点击窗口的“请选择需要识别的图片”按钮，从测试图片集中选择一张图片，再点击窗口的“运行”按钮，就会弹出识别到的训练集中的人脸照片。
# 实验结果
一共测试了10张人脸，识别的结果如下表：
![iamge2](https://github.com/githubhtz/image/blob/main/2.png)
![iamge3](https://github.com/githubhtz/image/blob/main/3.png)
![iamge4](https://github.com/githubhtz/image/blob/main/4.png)
可以看出识别成功率大约为80%，效果一般。
# Reference Code
1、https://blog.csdn.net/Just_Make_It_/article/details/105480090?utm_medium=distribute.pc_relevant_download.none-task-blog-BlogCommendFromBaidu-3.nonecase&depth_1-utm_source=distribute.pc_relevant_download.none-task-blog-BlogCommendFromBaidu-3.nonecas
2、https://github.com/coderwangson/Eigenface
3、https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis
4、https://github.com/shalouzaixiayu/tkinter-
