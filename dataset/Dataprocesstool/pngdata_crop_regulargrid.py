# name: Cao Yintao
# date: 2023/12/26 , 16:40
from PIL import Image
import os

def crop_image_to_grid(image_path, dest_folder):
    # 加载图像
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        cropsize = 256
        num_rows = img_height // cropsize
        num_cols = img_width // cropsize

        # 计算每个网格的尺寸
        crop_width = cropsize
        crop_height = cropsize

        # 创建保存裁剪图片的文件夹
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # 裁剪并保存图片
        count = 1
        for i in range(num_rows):
            for j in range(num_cols):
                # 计算裁剪区域
                left = j * crop_width
                top = i * crop_height
                right = (j + 1) * crop_width
                bottom = (i + 1) * crop_height
                crop_area = (left, top, right, bottom)

                # 裁剪并保存图片
                cropped_img = img.crop(crop_area)
                cropped_img.save(os.path.join(dest_folder, f"{count}.png"))
                print(f"{count}.png, saved to {dest_folder}")
                count += 1
#1.裁剪图片
# 使用示例
image_path = '/pj12/projectforwork/Process_data_bigimagecrop/Original_data_bigimage/1.png'  # 替换为您的图片路径
dest_folder = 'C:\cyt\pj1\pythonProject1\pj12\projectforwork\Process_data_bigimagecrop\Processed_data_smallimage\Image'         # 裁剪后图片保存的文件夹

crop_image_to_grid(image_path, dest_folder)

#2.裁剪Label
label_path = '/pj12/projectforwork/Process_data_bigimagecrop/Original_data_bigimage/3.png'  # 替换为您的图片路径
dest_folder_label = 'C:\cyt\pj1\pythonProject1\pj12\projectforwork\Process_data_bigimagecrop\Processed_data_smallimage\Label'         # 裁剪后图片保存的文件夹

crop_image_to_grid(label_path, dest_folder_label)
