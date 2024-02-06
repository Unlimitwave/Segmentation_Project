


#一、VOC示例数据集加载-图像分割训练模型步骤
1.数据集分割和数据集转化成VOC格式脚本在dataset文件夹里面，通过data_toVOCformat.py 脚本 完成

2.数据加载Dataloader在dataset文件夹里面, 已包含数据处理，裁剪，旋转功能，通过General_dataloader.py 脚本 完成

3.模型训练例如FCN模型，直接运行train_fcn.py即可

4.模型测试 运行modified-predict.py




****************************************************************************************************************

#二、GIS真实数据集加载（png格式）-图像分割训练模型步骤

1.PNG图片数据处理，将大张的Image.PNG图片和标签.png图片分割成小张图片，分割方法有网格法pngdata_crop_regulargrid.py，
滑动窗口方法pngdata_crop_slidewindow.py， 脚本在dataset下的Dataprocesstool文件夹里面
修改参数：cropsize, slide_step, 可能需要修改文件夹路径


2.数据增强，将分割后的小图片进行，翻转，旋转，椒盐噪声等处理，增加数据量， 通过 data_aug_adding.py脚本完成，
脚本在dataset下的Dataprocesstool文件夹里面
修改参数： 是否旋转，翻转加噪声， 可能需要修改文件路径


3.数据集转换成VOCformat数据集格式 通data_toVOCformat.py完成 ，
脚本在dataset下的Dataprocesstool文件夹里，
修改参数：可能需要修改文件路径

4.数据加载Dataloader在dataset文件夹里面, 已包含数据处理，裁剪，旋转功能，通过GIS_png_Dataloader.py 脚本 完成，
脚本在dataset下面
需要修改参数： cropsize， 文件路径，
如果数据类别标签有变化，一定要修改类别和标签。

5.模型训练例如FCN模型，train_fcn.py
脚本在projectforwork下面
需要import 相应的dataloader,例如GIS_png_Dataloader
可能需要import 相应的模型model，例如fcn_res101

需要修改参数：
（0）类别和标签
（1）超参数，batch, epoch, learning rate  等， 如果数据类别有变化，一定要修改class_num, 在训练代码里hist[n]的n要改，具体为n=class_num-1
另外如果数据的类别数目有变化，
一定要修改 util文件夹下的utils.py的class_num，
一定要修改model的输出通道数，例如fcn_res101模型的out_channel,
一定要修改eval.py 中的超参数class_num, 并且在eval.py  需要import 相应的dataloader,例如GIS_png_Dataloader， 以及验证集上需要设置数据集类型

(2)模型网络选择设置
net = ()?
例如net = fcn_res101.FCN()


(3)数据集Data类型和Dataloader选择设置
data_train = ?
例如data_train = GIS_png_Dataloader.GeneralDataset("train")

(4) 损失函数,优化器，学习率选择设置
例如
#损失
criterion = zl.CrossEntropy2d()
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
learning_rate = LR

(5)模型
加载已经训练过的模型
例如
 model = 'C:\\cyt\\pj1\\pythonProject1\\pj12\\projectforwork\\checkpoint\\train_fcn_2023-12-28_09-47-44_epoch-004_loss-0.3167_loss(val)-0.3102_acc-0.8711_miou-0.7520_miou(val)-0.7720.pth'


 6.如果要远程连接
 改一下训练脚本如train_unet.py 的model 模型加载，以及Dataloader 的prefix ，因为我用了绝对路径
