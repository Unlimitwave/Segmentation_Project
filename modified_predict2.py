# name: Cao Yintao
# date: 2024/1/2 , 10:25
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 这里需要替换成您实际使用的模型模块
import model.fcn_res101 as fcn_res101

# 颜色映射矩阵，每个类别一个颜色
COLOR_MAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128]
]


# 加载模型
def load_model(model_path):
    model = fcn_res101.FCN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# 将大图像分割成 512x512 像素的小块
def split_image(image_path):
    image = Image.open(image_path)
    image.resize((5632, 5632))
    width, height = image.size
    block_size = 512
    blocks = []

    for i in range(0, width, block_size):
        for j in range(0, height, block_size):
            box = (i, j, i + block_size, j + block_size)
            block = image.crop(box)
            blocks.append(block)
            print('split 1 block')

    return blocks


# 处理和预测每个小块
def process_and_predict(model, blocks):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []
    for block in blocks:
        block_tensor = transform(block).unsqueeze(0)  # 添加批次维度
        with torch.no_grad():
            output = model(block_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0)  # 去除批次维度
            predictions.append(prediction.cpu().numpy())
            print('1 block predicted')
    return predictions


# 解码分割图
def decode_segmentation(segmentation, color_map):
    color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(color_map):
        color_segmentation[segmentation == label] = color
        print('1 label colored')
    return color_segmentation


# 将小块重新组合成一张大图像
def reassemble_image(blocks, width=5632, height=5632, block_size=512):
    num_cols = width // block_size
    reassembled_image = np.zeros((height, width,3), dtype=np.uint8)

    for i, block in enumerate(blocks):
        row = i // num_cols
        col = i % num_cols
        reassembled_image[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size] = block
        print('1 block reassembeled')

    return reassembled_image


# 可视化图像
def visualize_image(image_array):
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.figure(figsize=(66.5, 66.5), dpi=100)

    # plt.axis('off')  # 不显示坐标轴
    plt.savefig('prediction_unet3.png', bbox_inches='tight', pad_inches=0)
    plt.show()


# 主要流程
def main(image_path, model_path):
    model = load_model(model_path)
    blocks = split_image(image_path)
    predictions = process_and_predict(model, blocks)
    color_predictions = [decode_segmentation(pred, COLOR_MAP) for pred in predictions]
    reassembled_image = reassemble_image(color_predictions)

    visualize_image(reassembled_image)

# 示例使用方式
model_path = 'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\Trained_model_pthfile\\trained_from_img1_model\\fcn_res101\\train_fcn_2023-12-29_13-56-48_epoch-017_loss-0.2188_loss(val)-0.2281_acc-0.9105_miou-0.8259_miou(val)-0.8306.pth'  # Set to actual model path
image_path = 'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\label_png\\label_2_json\\img.png' # Set to actual image path

main(image_path , model_path)

# 注意：需要提供实际的图像路径和模型路径才能运行此脚本。
