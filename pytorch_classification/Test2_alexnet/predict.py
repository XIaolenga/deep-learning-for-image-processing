###############预测模块          本节课里有训练自己的数据集
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(        
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])          #预处理，resize 、转化为张量、标准化处理

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)   #预处理
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0) # 扩充一个维度，加一个batch维度

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)#解码，解码成所需要使用的字典

    # create model
    model = AlexNet(num_classes=5).to(device)#####初始化网络

    # load model weights              载入训练好的模型
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()            #关闭掉dropout方法
    with torch.no_grad():  #不跟踪损失梯度
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()   ##将输出压缩，，，，将batch维度压缩掉
        predict = torch.softmax(output, dim=0)    ##获得概率分布
        predict_cla = torch.argmax(predict).numpy()## 获取概率最大处所得到的索引值

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

##打印类别名称和预测概率
if __name__ == '__main__':
    main()
