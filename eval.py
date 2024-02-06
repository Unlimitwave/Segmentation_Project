import torch
# import model.fcn_res101 as fcn_res101
import util.utils as tools
import dataset.pascal_data as pascal_data
import dataset.GIS_png_Dataloader as GIS_png_Dataloader
# import model.unet_123 as unet
import time
import os
import numpy as np
import util.zloss as zl


##############################################################::##########################################
                                # model.1验证集计算超参数设置  #
BATCH = 8
class_num = 9


# 对整个验证集进行计算
def eval_val(net, criterion=None, show_step=False, epoch=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ########################################################################################################
                                # model.2在验证集上需要设置数据集类型，eval.py  #
    data_val = GIS_png_Dataloader.GeneralDataset("val")
    # data_val=pascal_data.PASCAL_BSD("val")
    # data_val = cityspaces.CITYSPACES("val")

    ########################################################################################################
    val_data = torch.utils.data.DataLoader(data_val, batch_size=BATCH, shuffle=False)
    net = net.to(device)
    net = net.eval()

    if (criterion == None):
        criterion = zl.CrossEntropy2d()

    loss_all = 0
    acc = 0
    acc_cls = 0
    iou = 0
    miou = 0
    hist = np.zeros((class_num, class_num))
    st_epoch = time.time()
    for step, data in enumerate(val_data):
        st_step = time.time()
        img, img_gt = data
        img = img.to(device)
        img_gt = img_gt.to(device)

        with torch.no_grad():
            output = net(img)
            # 计算各项性能指标
            acc, acc_cls, iou, miou, hist = tools.get_MIoU(pred=output, label=img_gt, hist=hist)
            """
            label_true = img_gt.cpu().numpy()
            label_pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, miou, fwavacc = tools.label_accuracy_score(lbt, lbp, 21)
            """
            # 计算损失值
            loss = criterion(output, img_gt.long())
            loss_all = loss_all + loss.item()
            if (show_step == True):
                print(" 验证集：   (val)step[%d/%d]->loss:%.4f acc:%.4f miou:%.4f time:%ds   ------- 当前 epoch: %d" %
                      (step + 1, len(val_data), loss.item(), acc, miou, time.time() - st_epoch, epoch ))

    epoch_loss = loss_all / len(val_data)
    epoch_acc = acc
    epoch_miou = miou
    print("验证集：  当前epoch 评估指标：val->loss:%.4f acc:%.4f miou:%.4f time:%ds  ------- 当前 epoch: %d" %
         (epoch_loss, epoch_acc, epoch_miou, time.time() - st_epoch, epoch))

    with open("iou_eval.txt", "a") as f:
        f.write("epoch%d->" % (epoch) + str(iou) + "\n\n")

    # 保存hist矩阵
    Hist_path = "./pred_pic/epoch-%03d_val_hist.png" % (epoch)
    tools.drawHist(hist, Hist_path)

    return epoch_loss, epoch_acc, epoch_miou


# 将checkpoint文件夹中保存的模型都计算一遍
def eval_root():
    list_dir = os.listdir("./checkpoint")
    # net = PSPNet.PSPNet()
    # net = FCN.FCN()
    net = unet_123.Unet(21)
    max_miou = -1
    max_item = ""
    for item in list_dir:
        print(item)
        net.load_state_dict(torch.load("./checkpoint/" + item))
        epoch_loss, epoch_acc, epoch_miou = eval_val(net=net, show_step=False)
        if (max_miou < epoch_miou):
            max_miou = epoch_miou
            max_item = item
    print("max miou:%.4f item:%s" % (max_miou, max_item))


if __name__ == "__main__":
    eval_root()