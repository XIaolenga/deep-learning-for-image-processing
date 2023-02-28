import torch.nn as nn
import torch

# official pretrain weights提供的预训练权重
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):  
    def __init__(self, features, num_classes=1000, init_weights=False):                # 初始化，传入了特征提取模块，分类个数，是否进行网络权重初始化
        super(VGG, self).__init__()         
        self.features = features       # 特征提取模块        
        #####分类网络结构
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:                 #判断是否需要初始化
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)        #######全连接层前的展平操作，从第一个维度开始展平操作【batch，channel，high，weight】
        # N x 512*7*7
        x = self.classifier(x)      #传入分类网络模块
        return x    

    def _initialize_weights(self):             #权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):         #遍历网络每一层，若当前层为卷积层，采用凯明初始化
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) #遍历到偏置置零
            elif isinstance(m, nn.Linear):    #若为全连接层，进行Xavier初始化
                nn.init.xavier_uniform_(m.weight)   
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):  #特征提取模块           ，list类型，到时候只需要传入一个对应的配置列表就可以了
    layers = []                #定义一个空列表用来创建定义的每一层
    in_channels = 3            #3通道
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)#channel=3（第一层的），v=64（这一层的卷积核个数）
            layers += [conv2d, nn.ReLU(True)]  #卷积与rulu拼接
            in_channels = v
    return nn.Sequential(*layers)  ##将列表解包

#卷积层个数，M为池化层
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

##############实例化VGG网络配置模型
def vgg(model_name="vgg16", **kwargs):  #**表示可以传入可变长度的字典变量，实例化VGG16
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
