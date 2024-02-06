import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
# import Model123.PSPNet as PSPNet
# import Model123.DANet as DANet
# import Model123.fcn as FCN
# import Model123.fcn_res101 as fcn_res101
# import model.deeplabv3plus as deeplabv3plus
import model.pspnet.pspnet as pspnet
# import model.sync_batchnorm
# import model.sync_batchnorm.batchnorm
import util.utils as tools


# import dataset.pascal_data as pascal_data
import dataset.GIS_png_Dataloader as GIS_png_Dataloader
# import dataset.cityspaces as cityspaces
import eval
import time
import numpy as np
import matplotlib.pyplot as plt
import util.zloss as zl
from datetime import datetime
cur_datetime = datetime.now()
datetime_str = cur_datetime.strftime("%Y-%m-%d_%H-%M-%S")
script_name = os.path.basename(__file__)
model_name = script_name.split('.')[0]
# 各种标签所对应的颜色
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128]]

cm = np.array(colormap).astype("uint8")

########################################################################################################
                                         # 1.超参数设置 #


BATCH = 10
LR = 5e-5
EPOCHES =  200
class_num = 8
WEIGHT_DECAY = 1e-4
########################################################################################################

def train(offset, model, lr_update=False, show_img=False):
    # 加载网络
    ########################################################################################################
                                           #2.模型网络选择设置 #

    # net = PSPNet.PSPNet()
    # net = DANet.DANet()
    # net = FCN.fcn()
    # net = unet_123.Unet(21)
    net = pspnet.PSPNet(8, backbone='mobilenet',downsample_factor=16, aux_branch=False, pretrained=False)
    # ###################################################################################
    if (model != None):
        net.load_state_dict(torch.load(model))
        print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    ########################################################################################################
                                           # 3.数据集Data类型和Dataloader选择设置 #
    # 加载数据
    # data_train = pascal_data.PASCAL_BSD("train")
    # data_train = cityspaces.CITYSPACES("train")
    # data_val = pascal_data.PASCAL_BSD("val")
    data_train = GIS_png_Dataloader.GeneralDataset("train")



    train_data = torch.utils.data.DataLoader(data_train, batch_size=BATCH, shuffle=True)
    # val_data = torch.utils.data.DataLoader(data_val, batch_size=BATCH, shuffle=False)
    ########################################################################################################
                                           # 4.损失函数,优化器，学习率选择设置 #
    criterion = zl.CrossEntropy2d()
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=1e-4)
    learning_rate = LR
    # 开始训练
    print("开始训练(〃'▽'〃)")
    ########################################################################################################

    """
                                            1.训练(〃'▽'〃)循环代码：
    """
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
            print(hist[7])
            ''' 
            label_true = img_gt.cpu().numpy()
            label_pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, miou, fwavacc = tools.label_accuracy_score(lbt, lbp, 21)
            '''

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
            print("训练集：   step[%d/%d]->loss:%.4f acc:%.4f miou:%.4f lr:%.6f time:%ds ------- 当前 epoch: %d" %
                 (step + 1, len(train_data), loss.item(), acc, miou, learning_rate, time.time() - st_epoch, epoch))

        # print(iou)
        # print(hist)
        # 一个epoch训练完成，计算当前epoch数据
        epoch_loss = loss_all / len(train_data)
        epoch_acc = acc
        epoch_miou = miou
        print(np.diag(hist))
        # 打印信息
        print("训练集：  当前epoch 评估指标： epoch[%d/%d]->loss:%.4f acc:%.4f miou:%.4f lr:%.6f time:%ds  ------- 当前 epoch: %d" %
             (epoch+1, EPOCHES, epoch_loss, epoch_acc, epoch_miou, learning_rate, time.time() - st_epoch, epoch))

        #

        """
                                                   2.验证集Val计算：  验证集代码在eval.py(记得改下验证集的数据集类型)
        """
        ########################################################################################################
                                                 # model.在验证集上需要设置数据集类型，eval.py  #




        val_loss, val_acc, val_miou = eval.eval_val(net=net, criterion=criterion, epoch=epoch + offset, show_step=True)
        ########################################################################################################
        print(
            "验证集：  epoch[%d/%d],train_loss,%.4f,train_acc,%.4f,train_miou,%.4f,eval_loss,%.4f,eval_acc,%.4f,eval_miou,%.4f,lr,%.6f,time,%ds ------- 当前 epoch: %d" %
            (epoch + 1, EPOCHES, epoch_loss, epoch_acc, epoch_miou, val_loss, val_acc, val_miou, learning_rate,
             time.time() - st_epoch, epoch))

        # 保存当前训练数据


        """
                                                           3.模型保存
        """

        path = "./checkpoint/pspnet/%s,%s_epoch-%03d_loss-%.4f_loss(val)-%.4f_acc-%.4f_miou-%.4f_miou(val)-%.4f.pth" % \
               (model_name, datetime_str, epoch + offset, epoch_loss, val_loss, epoch_acc, epoch_miou, val_miou)
        torch.save(net.state_dict(), path)
        print("成功保存模型%s" % (path))

        with open("iou_train.txt", "a") as f:
            f.write("epoch%d->" % (epoch + offset) + str(iou) + "\n\n")

        with open("loss_train.txt", "a") as f:
            f.write("epoch%d->" % (epoch + offset) + str(epoch_loss) + "\n")

        # 保存hist矩阵
        Hist_path = "./pred_pic/epoch-%03d_train_hist.png" % (epoch + offset)
        tools.drawHist(hist, Hist_path)

        # 更新学习率
        if (lr_update == True):
            # 每20个epoch就将学习率降低10倍
            if (epoch + offset == 10):
                learning_rate = 1e-5
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=WEIGHT_DECAY)
                print("当前学习率lr=%.8f" % (learning_rate))
            if (epoch + offset == 80):
                learning_rate = learning_rate * 0.1
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=WEIGHT_DECAY)
                print("当前学习率lr=%.8f" % (learning_rate))

    return 0


if __name__ == "__main__":
    offset = 0
    model = None
    train(offset=offset, model=model, lr_update=True, show_img=False)