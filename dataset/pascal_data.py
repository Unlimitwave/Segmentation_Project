# pascal_data.py
import torch
import torchvision.transforms as tfs
import os
import scipy.io as scio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# PASCAL VOC语义分割增强数据集
prefix = "C:/cyt/pj1/pythonProject1/pj12/projectforwork/VOCdevkit/VOCdevkit/VOC2012/"

# 超参数，设置裁剪的尺寸
CROP = 256


class PASCAL_BSD(object):
    def __init__(self, mode="train", change=False):
        super(PASCAL_BSD, self).__init__()
        # 需要将标签和像素点颜色之间建立映射关系
        # 读取数据的模式, train 还是 val
        self.mode = mode
        # voc数据集对应类别标签，一共有20+1个类
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'potted plant',
                        'sheep', 'sofa', 'train', 'tv/monitor']
        # 颜色标签，分别对应21个类别
        self.colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                         [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                         [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                         [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                         [0, 192, 0], [128, 192, 0], [0, 64, 128]]

        # 将数据转换成tensor，并且做标准化处理
        self.im_tfs = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 将mat格式的数据转换成png格式
        if (change == True):
            self.mat2png()

        self.image_name = []
        self.label_name = []
        self.readImage()
        print("%s->成功加载%d张图片" % (self.mode, len(self.image_name)))

    """
    读取图片
    图片的名称在/ImageSets/Segmentation/train.txt ans val.txt里
    如果传入参数train为True，则读取train.txt的内容，否则读取val.txt的内容
    图片都在./data/VOC2012/JPEGImages文件夹下面，需要在train.txt读取的每一行后面加上.jpg
    标签都在./data/VOC2012/SegmentationClass文件夹下面，需要在读取的每一行后面加上.png
    最后返回记录图片路径的集合data和记录标签路径集合的label
    """

    # 读取图像和标签信息
    def readImage(self):
        img_root = prefix + "JPEGImages/"
        label_root = prefix + "SegmentationClass/"
        if (self.mode == "train"):
            with open(prefix + "ImageSets/Segmentation/train.txt", "r") as f:
                list_dir = f.readlines()
        elif (self.mode == "val"):
            with open(prefix + "ImageSets/Segmentation/val.txt", "r") as f:
                list_dir = f.readlines()
        for item in list_dir:
            self.image_name.append(img_root + item.split("\n")[0] + ".jpg")
            self.label_name.append(label_root + item.split("\n")[0] + ".png")

    # 数据处理，输入Image对象，返回tensor对象
    def data_process(self, img, img_gt):
        if (self.mode == "train"):
            # 以50%的概率左右翻转
            a = random.random()
            if (a > 0.5):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_gt = img_gt.transpose(Image.FLIP_LEFT_RIGHT)
            # 以50%的概率上下翻转
            a = random.random()
            if (a > 0.5):
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                img_gt = img_gt.transpose(Image.FLIP_TOP_BOTTOM)
            # 以50%的概率像素矩阵转置
            a = random.random()
            if (a > 0.5):
                img = img.transpose(Image.TRANSPOSE)
                img_gt = img_gt.transpose(Image.TRANSPOSE)
            a = random.random()
            # 进行随机裁剪
            width, height = img.size
            st = random.randint(0, 20)
            box = (st, st, width - 1, height - 1)
            img = img.crop(box)
            img_gt = img_gt.crop(box)

        img = img.resize((CROP, CROP))
        img_gt = img_gt.resize((CROP, CROP))

        img = self.im_tfs(img)
        img_gt = np.array(img_gt)
        img_gt = torch.from_numpy(img_gt)

        return img, img_gt

    def add_noise(self, img, gama=0.2):
        noise = torch.randn(img.shape[0], img.shape[1], img.shape[2])
        noise = noise * gama
        img = img + noise
        return img

    # 重载getitem函数，使类可以迭代
    def __getitem__(self, idx):
        # idx = 100
        img = Image.open(self.image_name[idx])
        img_gt = Image.open(self.label_name[idx])
        img, img_gt = self.data_process(img, img_gt)
        # img = self.add_noise(img)
        return img, img_gt

    def __len__(self):
        return len(self.image_name)

    # 将mat数据转换成png
    def mat2png(self, dataDir=None, outputDir=None):
        if (dataDir == None):
            dataroot = prefix + "cls/"
        else:
            dataroot = dataDir
        if (outputDir == None):
            outroot = prefix + "SegmentationClass/"
        else:
            outroot = outputDir
        list_dir = os.listdir(dataroot)
        for item in list_dir:
            matimg = scio.loadmat(dataroot + item)
            mattmp = matimg["GTcls"]["Segmentation"]
            # 将mat转换成png
            # print(mattmp[0][0])
            new_im = Image.fromarray(mattmp[0][0])
            print(outroot + item[:-4] + ".png")
            new_im.save(outroot + item[:-4] + ".png")


if __name__ == "__main__":
    data_train = PASCAL_BSD("train")
    print(data_train[300])

    data_val = PASCAL_BSD("val")
    train_data = torch.utils.data.DataLoader(data_train, batch_size=16, shuffle=True)
    val_data = torch.utils.data.DataLoader(data_val, batch_size=16, shuffle=False)
    for item in val_data:
        img, img_gt = item
        # print(img.shape)
        # print(img_gt.shape)
