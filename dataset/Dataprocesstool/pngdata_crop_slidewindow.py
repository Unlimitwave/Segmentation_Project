# name: Cao Yintao
# date: 2023/12/27 , 10:44
# name: Cao Yintao
# date: 2023/12/26 , 16:40
from PIL import Image
import os

def crop_image_to_sliding_window(source_folder, dest_folder):

    for file in os.listdir(source_folder):
        image_path = os.path.join(source_folder, file)
        file_name = os.path.splitext(file)[0]
        # 加载图像
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            cropsize = 512  # 窗口大小
            slide_step = 256  # 步长

            # 创建保存裁剪图片的文件夹
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # 裁剪并保存图片
            count = 1
            for y in range(0, img_height, slide_step):
                for x in range(0, img_width, slide_step):
                    # 计算裁剪区域
                    left = x
                    upper = y
                    if x + cropsize > img_width:
                        continue
                    else:
                        right = x + cropsize
                    if y + cropsize > img_height:
                        continue
                    else:
                        lower = y + cropsize

                    # 裁剪图片
                    crop_img = img.crop((left, upper, right, lower))
                    crop_img.save(os.path.join(dest_folder, f'{file_name}_crop_{count}.png'))
                    print(f'{file_name}_crop_{count}.png saved to {dest_folder}')

                    count += 1




#1.裁剪图片
# 使用示例
image_folder_path = '../../Process_data_bigimagecrop/Original_data_bigimage/All_images/Original_images' # 替换的图片路径

dest_folder = '../../Process_data_bigimagecrop/Processed_data_smallimage/Image'         # 裁剪后图片保存的文件夹

crop_image_to_sliding_window(image_folder_path, dest_folder)

#2.裁剪Label
label_folder_path = '../../Process_data_bigimagecrop/Original_data_bigimage/All_images/Original_lables'  # 替换为您的图片路径


dest_folder_label = '../../Process_data_bigimagecrop/Processed_data_smallimage/Label' # 裁剪后图片保存的文件夹

crop_image_to_sliding_window(label_folder_path, dest_folder_label)

