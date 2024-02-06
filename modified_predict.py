
# name: Cao Yintao
# date: 2023/12/21 , 13:45
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import model.fcn_res101 as fcn_res101  # 假设使用的是 FCN 模型
# import model.unet_123.unet3plus2 as unet3plus
# import model.unet as unet
# import model.segnet as segnet
import model.res_segnet as res_segnet
# 加载模型
def load_model(model_path):
    # model = fcn_res101.FCN()
    # model = unet3plus.UNet3Plus(3, 8)
    # model = unet.Unet(8)
    model = res_segnet.ResSegNet(9)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 处理图像
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((5632, 5632)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    print('Image sucessfully loaded')
    return transform(image).unsqueeze(0)  # 添加批次维度

# 进行预测
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)

        print('predict the model, output shape：', output.shape)
        predictions = torch.argmax(output, dim=1).squeeze()
        print('predict the model, prediction shape：',predictions.shape)
        return predictions

# 解码分割图
def decode_segmap(image, nc=9):
    label_colors = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 可视化预测结果
def visualize_prediction(prediction):
    process_prediction= prediction.cpu().numpy()
    print('prediction to numpy, numpy ndarry shape: ', type(process_prediction), process_prediction.shape)
    color_segmap = decode_segmap(process_prediction)
    print('numpy ndarry to color ndarry, color ndarry shape: ', type(color_segmap),color_segmap.shape)
    plt.figure(figsize=(66.5, 66.5),dpi=100)
    plt.imshow(color_segmap)
    # plt.axis('off')  # 不显示坐标轴
    plt.savefig('prediction_new_resseg.png', bbox_inches='tight', pad_inches=0)
    plt.show()


# 主函数
def main():
    model_path = 'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\Trained_model_pthfile_new9classes\\res_segnet\\train_res_segnet_2024-01-08_16-16-53_epoch-009_loss-0.2274_loss(val)-0.2318_acc-0.9223_miou-0.8891_miou(val)-0.8864.pth'  # Set to actual model path
    image_path = 'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\label_png\\label_2_json\\img.png' # Set to actual image path
    model = load_model(model_path)
    image_tensor = process_image(image_path)
    prediction = predict(model, image_tensor)
    visualize_prediction(prediction)

if __name__ == '__main__':
    main()

#'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\Trained_model_pthfile\\trained_from_img1_model\\fcn_res101\\train_fcn_2023-12-29_13-56-48_epoch-017_loss-0.2188_loss(val)-0.2281_acc-0.9105_miou-0.8259_miou(val)-0.8306.pth'
#'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\label_png\\label_2_json\\1.png'
#'C:\\cyt\\pj1\\pythonProject1\\pj12\\projectforwork\\Process_data_bigimagecrop\\Processed_data_smallimage\\Image\\1_crop_70.png'

