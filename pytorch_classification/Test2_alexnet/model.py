import torch.nn as nn
import torch

# 作者使用两块GPU进行并行运算
class AlexNet(nn.Module):#hhh
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(                              # 特征提取模块
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 55.25自动舍弃 若传入tuple：（1， 2）1代表上下各补一个0，2代表左右各补两列0
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(                            # 分类器模块（全连接层）
            nn.Dropout(p=0.5),                                      # 按照0.5的比例将全连接中的神经元进行失活，这样可以防止过拟合
            nn.Linear(128 * 6 * 6, 2048),                           #pytorch一般将batch放在首位
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:                                            #若传入参数init_weights=Ture ，那么将会调用初始化权重函数
            self._initialize_weights()

    def forward(self, x):                                           #前向传播
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):                                  
        for m in self.modules():                                   #创建一个迭代器
            if isinstance(m, nn.Conv2d):                           #判断所得到的层结构的类别’m‘是否属于卷积层，若是则进行凯初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')           #凯明初始化
                if m.bias is not None:                             #偏执不为空则赋0进行初始化
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):                         #若传入的实例为全链接层
                nn.init.normal_(m.weight, 0, 0.01)                 # 则通过正太分布对权重进行赋值操作（均值为0，方差为0.01）
                nn.init.constant_(m.bias, 0)                       # 偏执不为空则赋0进行初始化
