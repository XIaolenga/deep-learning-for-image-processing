from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


def _make_divisible(ch, divisor=8, min_ch=None):  # 将传入的Channel传入到离他最近的8的整数倍（训练速度提升）
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):           
    def __init__(self,
                 in_planes: int,        ## 输入特征矩阵的Channel
                 out_planes: int,         ## 输出特征矩阵的Channel
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,  #对应的卷积核接的BN层
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # 对应的就是激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:       # 如果没有传入norm_layer则默认采用BN层
            norm_layer = nn.BatchNorm2d
        if activation_layer is None: # 如果没有传入activation_layer则默认采用Relu6
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True)) ##卷积、BN层、激活函数三件套 


class SqueezeExcitation(nn.Module):  ### SE模块（注意力模块   相当于两个全连接）
    def __init__(self, input_c: int, squeeze_factor: int = 4):  # 第一个全连接层的节点个数是输入特征矩阵Channel的1/4 所以SqueezeExcitation 默认是等于4的
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8) # 1/4，把他调整到离他最近的8的整数倍
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:  # 这里传入的x为特征矩阵 返回Tensor格式
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  #平均池化成1*1的大小
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)    # 第二个全连接层采用Hardsigmoid函数
        return scale * x         ## 这里的scale相当于第二个全连接层所输出的数据了（相当于重要程度）            *x后得到通过注意力模块之后的输出


class InvertedResidualConfig:  ##Config文件对应的是MobileNetV3中的每一个bneck结构的参数配置文件
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,  # 对应这第一个1*1卷积层所使用的卷积核的个数
                 out_c: int,
                 use_se: bool,      ## 表示是否使用SE模块
                 activation: str,   #采用激活函数的类型（RELU 、 Hardswish）
                 stride: int,
                 width_multi: float):  # 这个参数为阿尔法 表示用来调节每个卷积层所使用channel的倍率因子
        self.input_c = self.adjust_channels(input_c, width_multi)  ## 先将input_c乘以阿尔法得到调节之后的input_c
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi) # 对应这第一个1*1卷积层所使用的卷积核的个数乘以阿尔法参数进行调节
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se  #是否使用SE模块
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod   #静态方法
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)# 调节到8的整数倍


class InvertedResidual(nn.Module):   ## 到残差模块
    def __init__(self,
                 cnf: InvertedResidualConfig,   #传入了上面所定义的Config文件   （InvertedResidualConfig）
                 norm_layer: Callable[..., nn.Module]): 
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:  # 判断步幅是否为1或者是2，如果不是则报错
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c) #是否使用shortcut连接

        layers: List[nn.Module] = []  # 定义一个空列表，每个元素为nn.model类型
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU   # 1.7版本以上

        # expand  对应的第一个1*1 的卷积层（升维）第一个bneck结构没有第一个1*1 的卷积层
        if cnf.expanded_c != cnf.input_c:   # ec不等于ic才有1*1 的卷积层
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,  # 输出
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,  # DW卷积的输入特征矩阵和输出特征矩阵的Channel是保持一致的
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))  #  线性激活

        self.block = nn.Sequential(*layers)  # block为整个到残差结构
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect: #判断是否使用shortcut连接
            result += x

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],  # 参数列表
                 last_channel: int,  # 倒数第二个全连接层的输出节点的个数
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,  # 到残差模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting: 
            raise ValueError("The inverted_residual_setting should not be empty.") # 判断列表是否为空
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):# 判断是不是列表
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:    ###  默认设置为BN
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer  第一个卷积结构
        firstconv_output_c = inverted_residual_setting[0].input_c   
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks  遍历每一个bneck结构，将参数传入
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c       # 获取最后一个bneck的输出
        lastconv_output_c = 6 * lastconv_input_c   # 960 = 160*6  676 = 96 * 6
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)   ### 主干网络（特征提取模块）
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:   #传入所需要的config文件即可
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
