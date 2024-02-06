import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import model.fcn_res101 as fcn_res101  # 假设使用的是 FCN 模型
import model.unet as unet
import re


# import model.unet_123.unet3plus2 as unet3plus
# import model.unet as unet
# 加载模型
def load_model(model_path):
    # model = fcn_res101.FCN()
    # model = unet3plus.UNet3Plus(3, 8)
    model = unet.Unet(8)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def split_image(image_path):
    image = Image.open(image_path)
    image = image.resize((5632, 5632))
    print('image size: ', image.size)
    width, height = image.size
    block_size = 512
    blocks = []
    count = 0
    for j in range(0, height, block_size):
        for i in range(0, width, block_size):
            box = (i, j, i + block_size, j + block_size)
            block = image.crop(box)
            blocks.append(block)
            print('split 1 block')
            block.save('./split_blockpic/count_%d.png' % count)
            count = count + 1

    return blocks





# 解码分割图
def decode_segmap(image, nc=8):
    label_colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128]
    ])

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






def process_and_predict_color(model, blocks):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    colored_predictions = []
    count = 0
    for block in blocks:
        block_tensor = transform(block).unsqueeze(0)  # 添加批次维度
        with torch.no_grad():
            output = model(block_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0)  # 去除批次维度
            prediction=prediction.cpu().numpy()
            print('prediction.shape', prediction.shape)
            print('1 block predicted')
            color_segmap = decode_segmap(prediction)
            print(type(color_segmap))
            print(color_segmap.shape)
            print('1 block colored %d' % count)
            plt.figure(figsize=(6.65, 6.65), dpi=100)
            plt.imshow(color_segmap)
            plt.axis('off')  # 不显示坐标轴
            plt.savefig('./colored_blockpic/%d_colored.png' % count, bbox_inches='tight', pad_inches=0)
            plt.close()
            colored_predictions.append(color_segmap)
            count = count +1
    return colored_predictions






# 将小块重新组合成一张大图像
def reassemble_image(folder_path, width=5632, height=5632, block_size=512):
    num_cols = width // block_size
    num_rows = height // block_size

    image_files = os.listdir(folder_path)
    def sort_key(filename):
        numbers = re.findall('\d+', filename)
        return [int(num) for num in numbers]


    image_files.sort(key=sort_key)
    large_image = Image.new('RGB', (width, height))
    for i, filename in enumerate(image_files):
        block_path = os.path.join(folder_path, filename)
        print(filename)
        with Image.open(block_path) as block:

            col = i % num_cols
            row = i // num_cols
            x_offset = col * block_size
            y_offset = row * block_size
            large_image.paste(block, (x_offset, y_offset))


    return large_image



# 可视化预测结果
def visualize_prediction(large_image):

    plt.figure(figsize=(77.93, 77.93), dpi=100)
    plt.imshow(large_image)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig('prediction_new.png', bbox_inches='tight', pad_inches=0)
    plt.show()


# 主函数
def main():
    model_path = 'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\Trained_model_pthfile\\trained_from_img1_model\\unet\\train_unet_2023-12-29_11-47-31_epoch-019_loss-0.2014_loss(val)-0.2306_acc-0.9178_miou-0.8417_miou(val)-0.8322.pth'  # Set to actual model path
    image_path = 'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\label_png\\label_2_json\\img.png'  # Set to actual image path
    model = load_model(model_path)
    blocks = split_image(image_path)
    colored_predictions= process_and_predict_color(model, blocks)
    large_image1 = reassemble_image('C:\\cyt\\pj1\\pythonProject1\\pj12\\projectforwork\\colored_blockpic')
    visualize_prediction(large_image1)



if __name__ == '__main__':
    main()

# 'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\Trained_model_pthfile\\trained_from_img1_model\\fcn_res101\\train_fcn_2023-12-29_13-56-48_epoch-017_loss-0.2188_loss(val)-0.2281_acc-0.9105_miou-0.8259_miou(val)-0.8306.pth'
# 'C:\\Users\\zkyt\\Desktop\\Projec-guideline\\GISimg1_project_work\\label_png\\label_2_json\\1.png'
# 'C:\\cyt\\pj1\\pythonProject1\\pj12\\projectforwork\\Process_data_bigimagecrop\\Processed_data_smallimage\\Image\\1_crop_70.png'

