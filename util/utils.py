# utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt

# 超参数，类别数量
#########################################################################
#类别变化时一定要更改
class_num = 9

##########################################################################
# 计算各种评价指标


# 计算混淆矩阵
def fast_hist(a, b, n):
    """
    生成混淆矩阵hist
    a 是形状为(HxW,)的预测标签值label_pred
    b 是形状为(HxW,)的真实标签值label_true
    n 是类别数
    """
    a = torch.softmax(a, dim=1)
    _, a = torch.max(a, dim=1)
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    # k为掩膜,在和b相对应的索引的位置上填入true或者false
    # b[k]会把mask中索引为true的元素输出
    # （去除了255这些点（即标签图中的白色的轮廓），其中的b>=0是为了防止bincount()函数出错）
    k = (b >= 0) & (b < n)
    hist = np.bincount(n * b[k].astype(int) + a[k].astype(int), minlength=n ** 2).reshape(n, n)
    # print(hist[20])
    return hist


def per_class_iou(hist):
    """
    hist传入混淆矩阵(n, n)
    """
    # 因为下面有除法，防止分母为0的情况报错
    np.seterr(divide="ignore", invalid="ignore")
    # 交集：np.diag取hist的对角线元素
    # 并集：hist.sum(1)和hist.sum(0)分别按两个维度相加，而对角线元素加了两次，因此减一次
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # 把报错设回来
    np.seterr(divide="warn", invalid="warn")
    # 如果分母为0，结果是nan，会影响后续处理，因此把nan都置为0
    iou[np.isnan(iou)] = 0.
    return iou


def per_class_acc(hist):
    """
    :param hist: 混淆矩阵
    :return: 每类的acc和平均的acc
    """
    np.seterr(divide="ignore", invalid="ignore")
    acc_cls = np.diag(hist) / hist.sum(1)
    np.seterr(divide="warn", invalid="warn")
    acc_cls[np.isnan(acc_cls)] = 0.
    return acc_cls


# 使用这个函数计算模型的各种性能指标
# 输入网络的输出值和标签值，得到计算结果
def get_MIoU(pred, label, hist):
    """
    :param pred: 预测向量
    :param label: 真实标签值
    :return: 准确率，每类的准确率，每类的iou, miou
    """
    hist = hist + fast_hist(pred, label, class_num)
    # print(hist[20])
    # 准确率
    acc = np.diag(hist).sum() / hist.sum()
    # 每类的准确率
    acc_cls = per_class_acc(hist)
    # 每类的iou
    iou = per_class_iou(hist)
    miou = np.nanmean(iou[1:])
    return acc, acc_cls, iou, miou, hist


# 更新学习率
def getNewLR(LR, net):
    LR = LR / 2
    print("更新学习率LR=%.6f" % LR)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    return optimizer, LR


# 绘制hist矩阵的可视化图并保存
def drawHist(hist, path):
    # print(hist)
    hist_ = hist[1:]
    hist_tmp = np.zeros((class_num - 1, class_num - 1))

    for i in range(len(hist_)):
        hist_tmp[i] = hist_[i][1:]

    # print(hist_tmp)
    hist = hist_tmp
    plt.matshow(hist)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.axis("off")
    plt.close()
    # plt.colorbar()
    # plt.show()
    if (path != None):
        plt.savefig(path)
        # print("%s保存成功" % path)


if __name__ == "__main__":
    # hist = np.random.randint(0, 20, size=(21, 21))
    drawHist(hist, None)