from torch import nn
import torch

### 论文中的bottleneck表示倒残差结构
def _make_divisible(ch, divisor=8, min_ch=None):  ###  make_divisible函数主要是为了让通道数等于8的倍数（可能是因为底层的硬件）
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor  
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)        ### 四舍五入
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):  ##卷积、BN层记忆RELU6的组合层   nn.sequential 不需要写forward函数
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):  ##group=1则表示为普通的卷积，若group=输入特征矩阵的深度（in_channel）
        padding = (kernel_size - 1) // 2  
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),    ##BN层不要偏置
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):      # 倒残差结构
    def __init__(self, in_channel, out_channel, stride, expand_ratio):## expand_ratio表示扩展因子（表格中的t）
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio    ## tk = k * t（输入特征矩阵的深度*扩展因子t）
        self.use_shortcut = stride == 1 and in_channel == out_channel  ## 判断是否使用捷径分支（当步距为1，且输入特征矩阵与输出特征矩阵的大小相同）

        layers = []
        if expand_ratio != 1:  #判断扩展因子是否为1，等于1则不要1*1的卷积
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv   DW卷积的输出特征矩阵的深度与输入特征矩阵的深度是相同的，若groups等于输入特征矩阵的深度则为DW卷积。
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)         1*1的普通卷积，采用的激活函数为线性激活函数，不能用上面定义的ConBNRelu层结构
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)  # 将上面所定义的层结构打包组合在一起called self.conv

    def forward(self, x):
        if self.use_shortcut:           ##判断是否使用捷径分支
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []             #定义特征提取模块
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:   # 遍历将特征提取模块导入网络
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization  权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x): 
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
