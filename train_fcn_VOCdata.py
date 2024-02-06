import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
# import model.PSPNet as PSPNet
# import model.DANet as DANet
# import model.fcn as FCN
import model.fcn_res101 as fcn_res101
import util.utils as tools
import dataset.pascal_data as pascal_data
# import dataset.cityspaces as cityspaces
import eval
import time
import numpy as np
import matplotlib.pyplot as plt
import util.zloss as zl

# 各种标签所对应的颜色
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

cm = np.array(colormap).astype("uint8")

#############
# 超参数设置 #
#############
BATCH = 10
LR = 5e-6
EPOCHES = 50
class_num = 21
WEIGHT_DECAY = 1e-4


def train(offset, model, lr_update=False, show_img=False):
    # 加载网络
    # net = PSPNet.PSPNet()
    # net = DANet.DANet()
    # net = FCN.fcn()
    net = fcn_res101.FCN()
    if (model != None):
        net.load_state_dict(torch.load(model))
        print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # 加载数据
    data_train = pascal_data.PASCAL_BSD("train")
    # data_train = cityspaces.CITYSPACES("train")
    # data_val = pascal_data.PASCAL_BSD("val")
    train_data = torch.utils.data.DataLoader(data_train, batch_size=BATCH, shuffle=True)
    # val_data = torch.utils.data.DataLoader(data_val, batch_size=BATCH, shuffle=False)
    # 损失函数
    criterion = zl.CrossEntropy2d()
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=1e-4)
    learning_rate = LR
    # 开始训练
    print("开始训练(〃'▽'〃)")
    for epoch in range(EPOCHES):
        # 总的损失值
        loss_all = 0
        # 评估的四个指标
        acc = 0
        acc_cls = 0
        iou = 0
        miou = 0
        hist = np.zeros((class_num, class_num))

        st_epoch = time.time()
        net = net.train()
        for step, data in enumerate(train_data):
            st_step = time.time()
            img, img_gt = data
            img = img.to(device)
            img_gt = img_gt.to(device)
            # 前向传播
            output = net(img)
            # 计算各项性能指标
            acc, acc_cls, iou, miou, hist = tools.get_MIoU(pred=output, label=img_gt, hist=hist)
            # print(hist[20])
            """
            label_true = img_gt.cpu().numpy()
            label_pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, miou, fwavacc = tools.label_accuracy_score(lbt, lbp, 21)
            """

            # 计算损失值
            loss = criterion(output, img_gt.long())
            loss_all = loss_all + loss.item()
            # 反向传播更新网络
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (show_img == True):
                plt.subplot(1, 3, 1), plt.imshow(img.cpu().detach()[0].permute(1, 2, 0).numpy()), plt.axis("off")
                plt.subplot(1, 3, 2), plt.imshow(cm[img_gt[0].detach().cpu().numpy()]), plt.axis("off")
                plt.subplot(1, 3, 3)
                _, idx = torch.max(torch.softmax(output, dim=1), dim=1)
                plt.imshow(cm[idx[0].cpu().detach().numpy()]), plt.axis("off")
                # plt.colorbar()
                plt.show()

            # 打印当前信息
            print("step[%d/%d]->loss:%.4f acc:%.4f miou:%.4f lr:%.6f time:%ds" %
                  (step + 1, len(train_data), loss.item(), acc, miou, learning_rate, time.time() - st_epoch))

        print(iou)
        # print(hist)
        # 一个epoch训练完成，计算当前epoch数据
        epoch_loss = loss_all / len(train_data)
        epoch_acc = acc
        epoch_miou = miou
        print(np.diag(hist))
        # 打印信息
        print("epoch[%d/%d]->loss:%.4f acc:%.4f miou:%.4f lr:%.6f time:%ds" %
              (epoch, len(train_data) - 1, epoch_loss, epoch_acc, epoch_miou, learning_rate, time.time() - st_epoch))

        # 在验证集上计算
        val_loss, val_acc, val_miou = eval.eval_val(net=net, criterion=criterion, epoch=epoch + offset)

        # 保存当前训练数据
        path = "./checkpoint/epoch-%03d_loss-%.4f_loss(val)-%.4f_acc-%.4f_miou-%.4f_miou(val)-%.4f.pth" % \
               (epoch + offset, epoch_loss, val_loss, epoch_acc, epoch_miou, val_miou)
        torch.save(net.state_dict(), path)
        print("成功保存模型%s✿✿ヽ(°▽°)ノ✿" % (path))

        with open("iou_train.txt", "a") as f:
            f.write("epoch%d->" % (epoch + offset) + str(iou) + "\n\n")

        # 保存hist矩阵
        Hist_path = "./pic/epoch-%03d_train_hist.png" % (epoch + offset)
        tools.drawHist(hist, Hist_path)

        # 更新学习率
        if (lr_update == True):
            # 每20个epoch就将学习率降低10倍
            if (epoch + offset == 10):
                learning_rate = 1e-5
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=WEIGHT_DECAY)
                print("当前学习率lr=%.8f" % (learning_rate))
            if (epoch + offset == 20):
                learning_rate = 5e-6
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=WEIGHT_DECAY)
                print("当前学习率lr=%.8f" % (learning_rate))

    return 0


if __name__ == "__main__":
    offset = 0
    model = None
    train(offset=offset, model=model, lr_update=False, show_img=False)