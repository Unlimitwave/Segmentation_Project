# name: Cao Yintao
# date: 2023/12/22 , 9:40
import os
import random
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


"""
此脚本，将创建VOC数据集格式的文件夹，将图片数据和图像分割的标签数据，转化成VOC数据集格式以便加载数据, 1.1 和1.2处更改路径即可

"""
# -------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   修改train_percent用于改变验证集的比例 9:1
#
#   当前该库将测试集当作验证集使用，不单独划分测试集
# -------------------------------------------------------#
trainval_percent = 1
train_percent = 0.9
# -------------------------------------------------------#
#  用于工作的数据目录Data
# -------------------------------------------------------#
Data_workdir_path = '/pj12/projectforwork/Data'


if __name__ == "__main__":

    ###############################################################################################
                                           #1.删除旧文件夹和内容并创建文件夹
    data_dest_folder = '../../Data'

    if os.path.exists(os.path.join(data_dest_folder, 'Data_VOCformat')):
        shutil.rmtree(os.path.join(data_dest_folder, 'Data_VOCformat'))
        print('Data_VOCformat文件夹存在，现在该文件夹及其内容已经被删除')
    else:
        print('Data_VOCformat文件夹不存在，自动创建该文件夹')

    os.makedirs(os.path.join(data_dest_folder, 'Data_VOCformat'), exist_ok=True)
    """
        1.1   需要更改文件路径，训练图片数据文件夹的路径， Replace with your actual JPG folder path 
    """

    src_images_folder = '../../Process_data_bigimagecrop/Processed_data_smallimage/Image'


    """
         1.2   需要更改文件路径，标签Label数据文件夹的路径， Replace with your actual JPG folder path     
    """

    src_labels_folder = '../../Process_data_bigimagecrop/Processed_data_smallimage/Label'


    dest_folder = '../../Data/Data_VOCformat'  # Replace with your desired destination folder path

    # Create the necessary folders in the destination
    os.makedirs(os.path.join(dest_folder, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'ImageSets', 'Segmentation'), exist_ok=True)

    # Copy all images from the JPG folder to the JPEGImages folder
    for file_name in os.listdir(src_images_folder):
        full_file_name = os.path.join(src_images_folder, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dest_folder, 'JPEGImages'))

    # Copy all label data from the Label folder to the SegmentationClass folder
    for file_name in os.listdir(src_labels_folder):
        full_file_name = os.path.join(src_labels_folder, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dest_folder, 'SegmentationClass'))

    print("图片数据成功复制到对应文件夹下")


    ################################################################################################
                            # 2.创建写入train.txt trainval.txt  val.txt等训练数据的名称

    VOCdevkit_path = '../../Data/Data_VOCformat'
    random.seed(0)
    print("准备生成 txt 在 ImageSets.")

    segfilepath = os.path.join(VOCdevkit_path, 'SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'ImageSets/Segmentation')

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

########################################################################################
    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums = np.zeros([256])
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。" % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，长度为%s, 不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png)),str(len(np.shape(png)))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。")
        else:
            print("标签图片%s的shape为%s，长度为%s, 属于灰度图或者八位彩图，PASS。" % (name, str(np.shape(png)),str(len(np.shape(png)))))
        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")