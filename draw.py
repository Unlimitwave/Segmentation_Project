import matplotlib.pyplot as plt

file = open('log.txt')  # 打开文档
lines = file.readlines()  # 读取文档数据
# epoch = list(1, range(len(lines))+1) #epoch可以直接赋值，不放心的就用下面epoch的代码
epoch = []
train_loss = []
val_loss = []
for line in lines[2:]:
    # split用于将每一行数据用自定义的符号（我用的是逗号）分割成多个对象
    # 取分割后的第0列，转换成float格式后添加到epoch列表中
    epoch.append(str(line.split(' ')[0]))
    # 取分割后的第2列，转换成float格式后添加到train_loss列表中
    train_loss.append(float(line.split(' ')[2]))
    # 取分割后的第8列，转换成float格式后添加到val_loss列表中
    val_loss.append(float(line.split(' ')[8]))
plt.figure()
plt.title('loss during training')  # 标题
plt.plot(epoch, train_loss, label="train_loss")
plt.plot(epoch, val_loss, label="valid_loss")
plt.legend()
plt.grid()
plt.show()