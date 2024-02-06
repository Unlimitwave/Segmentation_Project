# name: Cao Yintao
# date: 2023/12/21 , 13:45
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import model.fcn_res101 as fcn_res101  # 假设使用的是 PSPNet 模型

# 加载模型
def load_model(model_path):
    # 根据您的模型架构和训练代码，这里可能需要调整
    model = fcn_res101.FCN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 处理图像
def process_image(image_path):
    # 根据您的模型预处理需求进行调整
    #############################################################################

    #1.resize 大小

    ###########################################################################
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加批次维度

# 进行预测
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        # 使用 argmax 来获取每个位置的预测类别索引
        predictions = torch.argmax(output, dim=1).squeeze()  # 假设批次大小为1
        return predictions

# 主函数
def main():
    model = load_model('C:\\cyt\\pj1\\pythonProject1\\pj12\\projectforwork\\checkpoint\\fcn\train_fcn_2023-12-28_11-53-22_epoch-004_loss-0.2381_loss(val)-0.2401_acc-0.9037_miou-0.8147_miou(val)-0.8231.pth')  # 更改为模型的实际路径
    image_tensor = process_image('C:\\cyt\\pj1\\pythonProject1\\pj12\\projectforwork\\testpic1\\1.jpg')  # 更改为测试图像的实际路径
    prediction = predict(model, image_tensor)

    # 可视化预测结果（假设 prediction 是形状为 (256, 256) 的2维数组）
    plt.imshow(prediction.cpu().numpy(), cmap='gray')
    plt.show()
if __name__ == '__main__':
    main()


# 显示生成的测试脚本
