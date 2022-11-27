"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-23 15:42:00
 * @modify date 2022-01-07 19:43:33
 * @desc [description]
 """

import torch



class SavgDropTmaxCls(torch.nn.Module):

    def __init__(self, fc, dprob, num_segments, num_features, dim=1):
        super(SavgDropTmaxCls, self).__init__()
        
        self.dim = dim
        self.num_segments = num_segments
        self.num_features = num_features
        self.dprob = dprob

        self.module = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1), 
                        nn.Dropout(p=self.dprob)
                    )
        
        self.fc = fc

    def forward(self, input_tensor):

        nt, chw = input_tensor.size()

        h = int((chw/self.num_features)**0.5)

        # NTxCHW -> NTxCxHxW
        input_tensor = input_tensor.reshape(nt, self.num_features, h, h)

        # NTxCxHxW -> NTxCx1x1
        output = self.module(input_tensor)

        # NTxCx1x1 -> NTxC
        output = output.squeeze(3).squeeze(2)

        # NTxC -> NxTxC
        output = output.view((-1, self.num_segments) + output.size()[1:]) 
       
        # NxTxC -> NxC
        # output = F.adaptive_max_pool1d(output.permute(0,2,1), 1).squeeze()
        output = output.max(dim=self.dim, keepdim=False)[0]
       
        # NxC -> NxClass
        output = self.fc(output)
        
        return output


class TmaxDropSavgCls(torch.nn.Module):

    def __init__(self, fc, dprob, num_segments, num_features, dim=1):
        super(TmaxDropSavgCls, self).__init__()
        
        self.dim = dim
        self.num_segments = num_segments
        self.num_features = num_features
        self.dprob = dprob

        self.module = nn.Sequential(
                        nn.Dropout(p=self.dprob),
                        nn.AdaptiveAvgPool2d(1), 
                        nn.Flatten(),
                        fc
                    )


    def forward(self, input_tensor):

        nt, chw = input_tensor.size()

        h = int((chw/self.num_features)**0.5)

        # NTxCHW -> NxTxCxHxW
        input_tensor = input_tensor.reshape(-1, self.num_segments, self.num_features, h, h)

        # NxTxCxHxW -> NxCxHxW
        input_tensor = input_tensor.max(dim=self.dim, keepdim=False)[0]

        # NxCxHxW -> NxClass
        output = self.module(input_tensor)
        
        return output


import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, num_segments, num_classes, num_features, num_layers = 3, dim=1, height=7, dropout = 0.2):
        super(AttentionPooling, self).__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.num_clips = num_segments
        self.feature_dim = num_features
        self.num_layers = num_layers
        self.hidden_dim = num_features
        self.positional_encoding = PositionalEncoding(self.feature_dim, max_len=self.num_clips*height*height)
        
        # self.attention_pooling_layers = nn.ModuleList([nn.Linear(self.hidden_dim, int(self.hidden_dim/4), bias = False) for i in range(self.num_layers)])
        self.attention_pooling_layers = nn.ModuleList([nn.Linear(self.hidden_dim, 1, bias = False) for i in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.attention_encoder_layer = nn.Sequential(nn.Linear(self.hidden_dim, int(self.hidden_dim/4)), nn.ReLU())

        self.final_classifiers = nn.Linear(int(self.num_layers*self.hidden_dim/4), 2)
        

    def forward(self, features):

        nt, chw = features.size()

        h = int((chw/self.feature_dim)**0.5)

        # NTxCHW -> NxTHWxC
        features = features.reshape(-1, self.num_clips*h*h, self.feature_dim)

        features = self.positional_encoding(features)
        
        attended_features = []
        for i in range(self.num_layers):
            attention_pool_output = self.attention_pooling_layers[i](features)
            
            attention_weights = F.softmax(attention_pool_output.squeeze(-1), dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            attended_feature = torch.matmul(attention_weights.unsqueeze(1), features)

            attended_feature = self.attention_encoder_layer(attended_feature)
            
            attended_features.append(attended_feature.squeeze(1))

        final_features = torch.column_stack(attended_features)
        
        # NxF -> NxC
        final_outputs = self.final_classifiers(final_features)

        return final_outputs



import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        # x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        x = self.shift_basic(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)
    
    
    # CHANGE:  TorchScript JIT gets confused, incorrectly assuming all arguments are Tensors.  Fix with explicit typing.
    # def shift(x, n_segment, fold_div=3, inplace=False):
    @staticmethod
    def shift_basic(x, n_segment:int, fold_div:int = 3, inplace:bool = False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out_z = torch.zeros((n_batch, 1, fold, h, w))
            out_l = x[:, 1:][:,:, :fold]  # shift left
            out_r = x[:, :-1][:, :, fold: 2 * fold]  # shift right
            out_n = x[:, :, 2 * fold:]  # not shift

            out_l = torch.cat((out_l, out_z), dim = 1)
            out_r = torch.cat((out_z, out_r), dim = 1)

            out = torch.cat((out_l, out_r, out_n), dim = 2)

        return out.view(nt, c, h, w)
    
    
    # CHANGE:  TorchScript JIT gets confused, incorrectly assuming all arguments are Tensors.  Fix with explicit typing.
    # def shift(x, n_segment, fold_div=3, inplace=False):
    @staticmethod
    def shift(x, n_segment:int, fold_div:int = 3, inplace:bool = False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None

### ResNet Code ###



from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

# from .._internally_replaced_utils import load_state_dict_from_url
# from ..utils import _log_api_usage_once

try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        outplanes: int = None,
    ) -> None:
        super().__init__()
        if outplanes is None:
            outplanes = planes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, outplanes)
        self.bn2 = norm_layer(outplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class SegResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        num_seg_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        self.upsample_4 = Upsample(512, 256, 2, 2)
        self.up_layer4 = self._make_layer(block, 512, 512, layers[3], outplanes = 256, stride=1, dilate=replace_stride_with_dilation[2])
        self.upsample_3 = Upsample(256, 128, 2, 2)
        self.up_layer3 = self._make_layer(block, 256, 256, layers[2], outplanes = 128, stride=1, dilate=replace_stride_with_dilation[1])
        self.upsample_2 = Upsample(128, 64, 2, 2)
        self.up_layer2 = self._make_layer(block, 128, 128, layers[1], outplanes = 64, stride=1, dilate=replace_stride_with_dilation[0])
        # self.upsample_1 = Upsample(64, 32, 2, 2)
        self.upsample_1 = Upsample(64, 32, 4, 4)
        self.up_layer1 = self._make_layer(block, 32+1, 32+1, layers[0], outplanes = 16)

        self.seg_layer = nn.Conv2d(16, num_seg_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int,
        midplanes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        outplanes: int = None,
    ) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None

        if outplanes is None:
            outplanes = midplanes
        else:
            downsample = nn.Sequential(
                conv1x1(inplanes, outplanes, stride),
                norm_layer(outplanes),
            )

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != midplanes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, midplanes * block.expansion, stride),
                norm_layer(midplanes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes, midplanes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, outplanes
            )
        )
        # self.inplanes = planes * block.expansion
        inplanes=outplanes
        for idx in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    # planes,
                    # inplanes=outplanes,
                    planes=outplanes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    # outplanes=outplanes if idx == blocks-1 else None
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x_in: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        #Classification
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        #Segmentation
        s4 = self.upsample_4(x4)
        s4 = torch.cat([s4, x3], dim=1)
        s4 = self.up_layer4(s4)

        s3 = self.upsample_3(s4)
        s3 = torch.cat([s3, x2], dim=1)
        s3 = self.up_layer3(s3)

        s2 = self.upsample_2(s3)
        s2 = torch.cat([s2, x1], dim=1)
        s2 = self.up_layer2(s2)

        s1 = self.upsample_1(s2)
        s1 = torch.cat([s1, x_in], dim=1)
        s1 = self.up_layer1(s1)

        s = self.seg_layer(s1)

        return x, s

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> SegResNet:
    model = SegResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict = False) #NOTE - GRG : Changed load state dict strict to False
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SegResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


### TSM Code ###

class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

    elif isinstance(net, SegResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


#Code Adopted from: https://github.com/mit-han-lab/temporal-shift-module 

# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

import torchvision

from torch.nn.init import normal_, constant_

import torch.nn.functional as F

# Input dimensions are NxTxCxHxW
class TSN(nn.Module):
    def __init__(self, num_class, num_channels, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 st_consensus_type="SavgDropTavg", before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                #  is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 is_shift=True, shift_div=8, shift_place='block', fc_lr5=False,
                #  temporal_pool=False, non_local=False
                 ):
        super(TSN, self).__init__()

        self.num_class = num_class
        self.num_channels = num_channels
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.st_consensus_type = st_consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = False
        self.non_local = False

        if not before_softmax and st_consensus_type != "SavgDropTavg":
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 1 #5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        st_consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, st_consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
  
        setattr(self.base_model, self.base_model.last_layer_name, nn.Identity())

        new_fc = nn.Linear(feature_dim, num_class)
            
        std = 0.001
        if hasattr(new_fc, 'weight'):
            normal_(new_fc.weight, 0, std)
            constant_(new_fc.bias, 0)

        if self.base_model_name == "resnet18":
            num_features = self.base_model.layer4[0].downsample[1].num_features
        else: 
            raise ValueError(f"Undefined num_features for base model = {self.base_model_name}!")

        #Define space time aggregration layer 
        if self.st_consensus_type == "VideoTransformer":
            self.consensus = st.VideoTransformer(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "VideoTransformer2":
            self.consensus = st.VideoTransformer2(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "TmaxSavgDropCls":
            self.consensus = st.TmaxSavgDropCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "TmaxDropSavgCls":
            self.consensus = TmaxDropSavgCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "SavgDropTmaxCls":
            self.consensus = SavgDropTmaxCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "SavgDropClsTavg":
            self.consensus = st.SavgDropClsTavg(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        else:
            raise ValueError(f"Unsupported cfg.ST_CONCENSUS = {self.st_consensus_type}!")

        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            
            #GRG: Change channels to 1 for grey image input
            # self.base_model.conv1.in_channels = 1
            self.base_model.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            if self.is_shift:
                print('Adding temporal shift...')
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                from model.video_models.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            # self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.base_model.avgpool = nn.Identity()

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from model.video_models.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            from model.video_models.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable
    
    
    
    def preprocess_input(self, video_frames):

         #Conver to rect image
        left_upper_point, left_lower_point, right_upper_point, right_lower_point = [[795.1225906658183, 204.40587222575687], [411.12618361465627, 762.2111793106028], [1113.0985628204648, 203.0585164115423], [1490.358190800554, 766.2532467532467]]

        # rct_src_i, rct_src_j = get_params_cvt_sector_to_rectangle(video_frames[0], left_upper_point, right_upper_point, left_lower_point, right_lower_point)

        # video_frames = [cvt_sector_to_rectangle(f, rct_src_i, rct_src_j) for f in video_frames]

        # find the center of the arc
        k_left = (left_lower_point[1] - left_upper_point[1]) / (left_lower_point[0] - left_upper_point[0])
        k_right = (right_lower_point[1] - right_upper_point[1]) / (right_lower_point[0] - right_upper_point[0])

        center_x = k_left * left_upper_point[0] - left_upper_point[1] - k_right * right_upper_point[0] + right_upper_point[1]
        center_x /= (k_left - k_right)

        center_y = k_left * center_x - k_left * left_upper_point[0] + left_upper_point[1]

        # find the radium
        r_upper = np.sqrt((center_x - left_upper_point[0]) ** 2 + (center_y - left_upper_point[1]) ** 2)
        r_lower = np.sqrt((center_x - left_lower_point[0]) ** 2 + (center_y - left_lower_point[1]) ** 2)

        # find the angle
        theta = 2 * np.arcsin((right_upper_point[0] - left_upper_point[0]) / 2 / r_upper)
        width = 2 * theta * (r_lower + r_upper) / 2
        height = r_lower - r_upper

        height = round(height)
        width = round(width)

        # interpolate
        j, i = np.meshgrid(np.arange(width), np.arange(height))

        r = i + r_upper
        t = theta / width * j
        src_x = center_x + np.sin(t - theta / 2) * r
        src_y = center_y + np.cos(t - theta / 2) * r

        src_j = np.round(src_x).astype(int)
        src_i = np.round(src_y).astype(int)
        # src_j = src_x
        # src_i = src_y
        
        src_j = src_j.clip(min = 0.0, max = float(video_frames.shape[-1]) - 1)
        src_i = src_i.clip(min = 0.0, max = float(video_frames.shape[-2]) - 1)
        
        video_frames = video_frames[:,:,:,src_i, src_j]

        #Resize frames
        image_size = (224, 224)
        
        # video_frames = [cv2.resize(f, image_size) for f in video_frames]
        
        n, t, c, h, w = video_frames.shape
        video_frames = video_frames.reshape(-1, c, h, w)
        # video_frames = F.interpolate(video_frames, size = image_size, mode = "bicubic", align_corners = True)
        # video_frames = F.interpolate(video_frames, size = (1, 224, 224), mode = "trilinear", align_corners = True)
        video_frames = F.interpolate(video_frames, size = image_size, mode = "bilinear", align_corners = True)
        
        video_frames = video_frames.reshape(n, t, c, image_size[0], image_size[1])

        # # clip = (clip - clip.min())/(clip.max() - clip.min())
        video_frames = video_frames.clamp(min=0.0, max=255.0) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)

        video_frames = video_frames/255.0

        return video_frames
    
    
    
    # CHANGE:  TorchScript JIT gets confused, incorrectly assuming all arguments are Tensors.  Fix with explicit typing.
    # def forward(self, input, no_reshape=False):
    def forward(self, input, no_reshape:bool = False):
        #Preprocess input
        input = self.preprocess_input(input)
        
        if not no_reshape:
            # sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
            
            # #GRG: Change channels to 1 for grey image input
            # sample_len = (1 if self.modality == "RGB" else 2) * self.new_length

            # if self.modality == 'RGBDiff':
            #     sample_len = 3 * self.new_length
            #     input = self._get_diff(input)

            # base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
            base_out = self.base_model(input.view((-1,) + input.size()[-3:]))
        else:
            base_out = self.base_model(input)


        output = self.consensus(base_out)

        return  {"pred_cls": output, "input": input}

    def seqForward(self, x, feature_module, feature_extractor):
        
        reshape = True
        if reshape:
            # sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
            
            #GRG: Change channels to 1 for grey image input
            sample_len = (1 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                x = self._get_diff(x)

            x = x.view((-1,) + x.size()[-3:])

        for name, module in self.base_model._modules.items():
            if module == feature_module:
                target_activations, x = feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        x = self.consensus(x)

        target_activations[0] = target_activations[0].view((-1, self.num_segments) + target_activations[0].size()[1:])
               
        return target_activations, x


### TSM Seg Code ###


class TSNseg(nn.Module):
    def __init__(self, num_class, num_seg_class, num_channels, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 st_consensus_type="SavgDropTavg", before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                #  is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 is_shift=True, shift_div=8, shift_place='block', fc_lr5=False,
                #  temporal_pool=False, non_local=False
                 ):
        super(TSNseg, self).__init__()

        self.num_class = num_class
        self.num_seg_class = num_seg_class
        self.num_channels = num_channels
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.st_consensus_type = st_consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = False
        self.non_local = False

        if not before_softmax and st_consensus_type != "SavgDropTavg":
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 1 #5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN Segmentation with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        st_consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, st_consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
  
        setattr(self.base_model, self.base_model.last_layer_name, nn.Identity())

        new_fc = nn.Linear(feature_dim, num_class)
            
        std = 0.001
        if hasattr(new_fc, 'weight'):
            normal_(new_fc.weight, 0, std)
            constant_(new_fc.bias, 0)

        if self.base_model_name == "resnet18":
            num_features = self.base_model.layer4[0].downsample[1].num_features
        else: 
            raise ValueError(f"Undefined num_features for base model = {self.base_model_name}!")

        #Define space time aggregration layer 
        if self.st_consensus_type == "VideoTransformer":
            self.consensus = st.VideoTransformer(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "VideoTransformer2":
            self.consensus = st.VideoTransformer2(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "AttentionPooling":
            self.consensus = AttentionPooling(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "TmaxSavgDropCls":
            self.consensus = st.TmaxSavgDropCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "TmaxDropSavgCls":
            self.consensus = TmaxDropSavgCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "SavgDropTmaxCls":
            self.consensus = SavgDropTmaxCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "SavgDropClsTavg":
            self.consensus = st.SavgDropClsTavg(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        else:
            raise ValueError(f"Unsupported cfg.ST_CONCENSUS = {self.st_consensus_type}!")

        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            # self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            
            # from model import resnet_seg
            # # self.base_model = getattr(model.resnet_seg, base_model)(True if self.pretrain == 'imagenet' else False)
            # self.base_model = getattr(resnet_seg, base_model)(True if self.pretrain == 'imagenet' else False, num_seg_classes = self.num_seg_class)
            self.base_model = globals()[base_model](True if self.pretrain == 'imagenet' else False, num_seg_classes = self.num_seg_class)
            
            #GRG: Change channels to 1 for grey image input
            # self.base_model.conv1.in_channels = 1
            self.base_model.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            if self.is_shift:
                print('Adding temporal shift...')
                # from model.video_models.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                from model.video_models.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            # self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.base_model.avgpool = nn.Identity()

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from model.video_models.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from model.video_models.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            from model.video_models.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSNseg, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def preprocess_input(self, video_frames):

         #Conver to rect image
        left_upper_point, left_lower_point, right_upper_point, right_lower_point = [[795.1225906658183, 204.40587222575687], [411.12618361465627, 762.2111793106028], [1113.0985628204648, 203.0585164115423], [1490.358190800554, 766.2532467532467]]

        # rct_src_i, rct_src_j = get_params_cvt_sector_to_rectangle(video_frames[0], left_upper_point, right_upper_point, left_lower_point, right_lower_point)

        # video_frames = [cvt_sector_to_rectangle(f, rct_src_i, rct_src_j) for f in video_frames]

        # find the center of the arc
        k_left = (left_lower_point[1] - left_upper_point[1]) / (left_lower_point[0] - left_upper_point[0])
        k_right = (right_lower_point[1] - right_upper_point[1]) / (right_lower_point[0] - right_upper_point[0])

        center_x = k_left * left_upper_point[0] - left_upper_point[1] - k_right * right_upper_point[0] + right_upper_point[1]
        center_x /= (k_left - k_right)

        center_y = k_left * center_x - k_left * left_upper_point[0] + left_upper_point[1]

        # find the radium
        r_upper = np.sqrt((center_x - left_upper_point[0]) ** 2 + (center_y - left_upper_point[1]) ** 2)
        r_lower = np.sqrt((center_x - left_lower_point[0]) ** 2 + (center_y - left_lower_point[1]) ** 2)

        # find the angle
        theta = 2 * np.arcsin((right_upper_point[0] - left_upper_point[0]) / 2 / r_upper)
        width = 2 * theta * (r_lower + r_upper) / 2
        height = r_lower - r_upper

        height = round(height)
        width = round(width)

        # interpolate
        j, i = np.meshgrid(np.arange(width), np.arange(height))

        r = i + r_upper
        t = theta / width * j
        src_x = center_x + np.sin(t - theta / 2) * r
        src_y = center_y + np.cos(t - theta / 2) * r

        src_j = np.round(src_x).astype(int)
        src_i = np.round(src_y).astype(int)
        # src_j = src_x
        # src_i = src_y
        
        src_j = src_j.clip(min = 0.0, max = float(video_frames.shape[-1]) - 1)
        src_i = src_i.clip(min = 0.0, max = float(video_frames.shape[-2]) - 1)
        
        video_frames = video_frames[:,:,:,src_i, src_j]

        #Resize frames
        image_size = (224, 224)
        
        # video_frames = [cv2.resize(f, image_size) for f in video_frames]
        
        n, t, c, h, w = video_frames.shape
        video_frames = video_frames.reshape(-1, c, h, w)
        # video_frames = F.interpolate(video_frames, size = image_size, mode = "bicubic", align_corners = True)
        # video_frames = F.interpolate(video_frames, size = (1, 224, 224), mode = "trilinear", align_corners = True)
        video_frames = F.interpolate(video_frames, size = image_size, mode = "bilinear", align_corners = True)
        
        video_frames = video_frames.reshape(n, t, c, image_size[0], image_size[1])

        # # clip = (clip - clip.min())/(clip.max() - clip.min())
        video_frames = video_frames.clamp(min=0.0, max=255.0) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)

        video_frames = video_frames/255.0

        return video_frames
    
    
    
    # CHANGE:  TorchScript JIT gets confused, incorrectly assuming all arguments are Tensors.  Fix with explicit typing.
    # def forward(self, input, no_reshape=False):
    def forward(self, input, no_reshape:bool = False):
        #Preprocess input
        input = self.preprocess_input(input)

        if not no_reshape:
            base_out, seg_out = self.base_model(input.view((-1,) + input.size()[-3:]))
        else:
            base_out, seg_out = self.base_model(input)


        output = self.consensus(base_out)

        nt, c, h, w = seg_out.shape
        seg_out = seg_out.reshape(-1, self.num_segments, c, h, w)

        return {"pred_cls": output, "pred_seg": seg_out, "input": input}

    def seqForward(self, x, feature_module, feature_extractor):
        
        reshape = True
        if reshape:
            # sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
            
            #GRG: Change channels to 1 for grey image input
            sample_len = (1 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                x = self._get_diff(x)

            x = x.view((-1,) + x.size()[-3:])

        for name, module in self.base_model._modules.items():
            
            #Don't run gradCAM through segmentation branch
            if "up" in name.lower():
                break

            if module == feature_module:
                target_activations, x = feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        x = self.consensus(x)

        target_activations[0] = target_activations[0].view((-1, self.num_segments) + target_activations[0].size()[1:])
               
        return target_activations, x


### Conver to Rect ###

import numpy as np

from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, NearestNDInterpolator

def get_params_cvt_sector_to_rectangle(image, left_upper_point, right_upper_point, left_lower_point, right_lower_point):
    """
    convert a sector image to a rectangle
    """
    
    # find the center of the arc
    k_left = (left_lower_point[1] - left_upper_point[1]) / (left_lower_point[0] - left_upper_point[0])
    k_right = (right_lower_point[1] - right_upper_point[1]) / (right_lower_point[0] - right_upper_point[0])

    center_x = k_left * left_upper_point[0] - left_upper_point[1] - k_right * right_upper_point[0] + right_upper_point[1]
    center_x /= (k_left - k_right)

    center_y = k_left * center_x - k_left * left_upper_point[0] + left_upper_point[1]

    # find the radium
    r_upper = np.sqrt((center_x - left_upper_point[0]) ** 2 + (center_y - left_upper_point[1]) ** 2)
    r_lower = np.sqrt((center_x - left_lower_point[0]) ** 2 + (center_y - left_lower_point[1]) ** 2)

    # find the angle
    theta = 2 * np.arcsin((right_upper_point[0] - left_upper_point[0]) / 2 / r_upper)
    width = 2 * theta * (r_lower + r_upper) / 2
    height = r_lower - r_upper

    height = round(height)
    width = round(width)

    # interpolate
    j, i = np.meshgrid(np.arange(width), np.arange(height))

    r = i + r_upper
    t = theta / width * j
    src_x = center_x + np.sin(t - theta / 2) * r
    src_y = center_y + np.cos(t - theta / 2) * r

    # src_j = np.round(src_x).astype(int)
    # src_i = np.round(src_y).astype(int)
    src_j = src_x
    src_i = src_y
    
    src_j = src_j.clip(min = 0, max = image.shape[1] - 1)
    src_i = src_i.clip(min = 0, max = image.shape[0] - 1)

    return src_i, src_j

def cvt_sector_to_rectangle(image, src_i, src_j):

    # new_image = []
    # for c in range(image.shape[-1]):
    #     img_spline = RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), image[:, :, c])
    #     new_image.append(img_spline.ev(src_i, src_j))
    # # new_image = image[src_i, src_j]
    
    # new_image = np.array(new_image).transpose(1,2,0)
    
    print('    creating spline')
    img_spline = RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), image)
    print('    applying spline')
    new_image = img_spline.ev(src_i, src_j)
    # img_spline = RegularGridInterpolator((np.arange(image.shape[0]), np.arange(image.shape[1])), image, method = "nearest")
    # new_image = img_spline((src_i, src_j))
    # img_spline = NearestNDInterpolator(list(zip(np.arange(image.shape[0]), np.arange(image.shape[1]))), image)
    # new_image = img_spline(src_i, src_j)
    print('    returning image')
    return new_image

import cv2

def load_images(file_name, x_range=None, y_range=None):
    cap = cv2.VideoCapture(file_name)
    print(file_name)
    images = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        # print(frame)
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # if x_range != None and y_range != None:
            # gray = gray[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            images.append(gray)
        else:
            break
    cap.release()
    return images


def saveSegPredAsGIF(clip, seg_pred, pred, path, filename):

    txt_ht = 80

    (img_height, img_width,) = clip.shape[-3:-1]

    
        
    image_list = []
    for frame_no, (clip_f, seg_pred_f) in enumerate(zip(clip, seg_pred)):

        # frame = np.tile(frame, (1,1,3))

        # #Convert img from BGR to RGB color format
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame = (frame*255.0).astype(np.uint8)


        result_img = getTextImg(txt_ht, img_width*2, frame_no, None, pred, filename)

        r1 = np.concatenate((clip_f, seg_pred_f), axis = 1)

        frame = np.concatenate((result_img, r1), axis = 0)
        # plt.imshow(frame)
        # plt.show()
        image_list.append(frame)

    saveAsVideo(image_list, path)

    return image_list


def getTextImg(txt_ht, img_width, frame_no = None, target_activities = None, predicted_activities = None, video_name = None):

    font_scale = 0.4
    y_place = 5
    x_place = 12

    result_img = np.zeros((txt_ht, img_width, 3), np.uint8)
    
    #Add video name
    if video_name is not None:
        cv2.putText(result_img, f"{video_name}",
                    (y_place,x_place), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 255, 255), 
                    1)

        x_place += 20
    
    #Add frame no and avg IoU
    if frame_no is not None:
        cv2.putText(result_img, f"Frame No.: {frame_no}",
                (y_place,x_place), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 255, 255), 
                1)

        x_place += 20
    
    
    #Add frame no and avg IoU
    if target_activities is not None:
        cv2.putText(result_img, f"Target: {target_activities}",
                (y_place,x_place), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 255, 255), 
                1)

        x_place += 20
    
    #Add frame no and avg IoU
    if predicted_activities is not None:
        cv2.putText(result_img, f"Prediction: {predicted_activities}",
                (y_place,x_place),  
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 255, 255), 
                1)

    return result_img


import imageio


def saveAsVideo(image_list, video_name='temp.gif'):
    # # out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps = 15, frameSize = len(image_list))
    # out = cv2.VideoWriter(filename = video_name, apiPreference = cv2.CAP_FFMPEG,  fourcc = cv2.VideoWriter_fourcc(*'DIVX'), fps = 1, frameSize = image_list[0].shape[:2])
    # for img in image_list:
    #     out.write(img)
    # out.release()
    imageio.mimsave(video_name, image_list)


colormap = [
                (0, 0, 0), #0 #000000 - Background
                (242, 5, 246), #1 #F205F6 - pleural line pneumothorax
                (255, 0, 0), #2 #FF0000 - pleural line normal
                (0, 255, 0), #3 #00FF00 - vessel
                (42, 125, 209), #4 #2A7DD1 - chest wall muscle
                # 4: (42, 135, 209), #4 #2A7DD1
                (221, 255, 51), #5 #DDFF33 - rib bone
                # 5: (209, 135, 42), #5
            ]
colormap = np.array(colormap, dtype = np.uint8)


import os

if __name__ == "__main__":

    # CHANGE
    JOHN_SYS = False


    SEG = True

    if JOHN_SYS:
        #model_path = "/Users/jgaleotti/Research/POCUS_AI_DARPA_src/PTX/v1/tsm_DARPA-Dataset_Rct_SavgDropTmaxCls_T7_A_epoch=9-step=79.ckpt"

        model_path = "tsm_DARPA-Dataset_Rct_SavgDropTmaxCls_T7_A_epoch=9-step=79.ckpt"

        if SEG:
            model_path = "tsm_seg_DARPA-Seg-Dataset_Seg_SavgDropTmaxCls_T3_A_epoch=15-step=127.ckpt"

        video_path = "image_1394469579519_clean.mp4"
    else:
        model_path = "/home/grg/Research/DARPA-Pneumothorax/results/Pneumothorax_Exps/tsm_DARPA-Dataset_Rct_SavgDropTmaxCls_T7_A/Pneumothorax_Exps_Pneumothorax_Exps/1dn54sdi_0/checkpoints/epoch=9-step=79.ckpt"
    
        if SEG:
            model_path = "/home/grg/Research/DARPA-Pneumothorax/results/Pneumothorax_Exps/tsm_seg_DARPA-Seg-Dataset_Seg_SavgDropTmaxCls_T3_A/Pneumothorax_Exps_Pneumothorax_Exps/8q6efize_0/checkpoints/epoch=15-step=127.ckpt"
            
        video_path = "/data1/datasets/DARPA-Dataset/NoSliding/image_1394469579519_clean.mp4"
    
    # Initialize model
    # Input dimensions are NxTxCxHxW = 1x15x1x224x224 (with rectification TxHxW = 15x1080x1920)
    
    
    if SEG:
        model = TSNseg(
                        num_class = 2, 
                        num_seg_class = 6, 
                        num_channels = 1,
                        num_segments = 15, 
                        modality = "RGB",
                        base_model = "resnet18",
                        pretrain = "imagenet",
                        is_shift = True, 
                        shift_div = 8, 
                        shift_place = 'blockres', #'block',
                        partial_bn = False,
                        dropout = 0.5,
                        st_consensus_type = "SavgDropTmaxCls"
                    )
    else:
        model = TSN(
                        num_class = 2, 
                        num_channels = 1,
                        num_segments = 15, 
                        modality = "RGB",
                        base_model = "resnet18",
                        pretrain = "imagenet",
                        is_shift = True, #False, #True,
                        shift_div = 8, 
                        shift_place = 'blockres', #'block',
                        partial_bn = False,
                        dropout = 0.5,
                        st_consensus_type = "SavgDropTmaxCls"
                    )

        

    #Load model weights
    # CHANGE
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    #Fix keys
    state_dict = {}
    for k,v in checkpoint["state_dict"].items():
        if not model.is_shift:
            k = k.replace(".net", "")
        state_dict[".".join(k.split(".")[1:])] = v
    
    model.load_state_dict(state_dict)
    
    # Put network in inference (evaluation) mode
    model.eval()
    
    
    #Load Video
    print('loading video')

    video_frames = load_images(video_path)
    print(len(video_frames), 'frames loaded')
    
    # # Speed up by processing only first 15 frames (single inference call)
    # video_frames = video_frames[0:15]

#    #Conver to rect image
#    print('preparing to rectify the video')
#    left_upper_point, left_lower_point, right_upper_point, right_lower_point = [[795.1225906658183, 204.40587222575687], [411.12618361465627, 762.2111793106028], [1113.0985628204648, 203.0585164115423], [1490.358190800554, 766.2532467532467]]
#
#    rct_src_i, rct_src_j = get_params_cvt_sector_to_rectangle(video_frames[0], left_upper_point, right_upper_point, left_lower_point, right_lower_point)
#    print('rectifying video')
#    video_frames = [cvt_sector_to_rectangle(f, rct_src_i, rct_src_j) for f in video_frames]
#
#    #Resize frames
#    print('resizing video')
#    image_size = (224, 224)
#
#    video_frames = [cv2.resize(f, image_size) for f in video_frames]
    
    
    #Model Forward pass
    print('Evaluating model')
    class_labels = [ 'no-sliding', 'sliding', ]
    
    idx_count = int(len(video_frames)/model.num_segments)
    print('idx_count = ', idx_count, ', frame count is ', len(video_frames), ', segment count is ', model.num_segments)
    with torch.no_grad():
        
        for idx in range(idx_count):
            print('    preparing frame ', idx)
            clip = video_frames[idx*model.num_segments:(idx+1)*(model.num_segments)]
            
            # Remap clip to have dimensions NxTxCxHxW:
            clip = torch.tensor(clip).unsqueeze(1).unsqueeze(0).float()
            
            print('    evaluating frame ', idx)
            out_dict = model(clip)

            pred = out_dict["pred_cls"]

            pred = F.softmax(pred)

            print(f"[{idx}] Model prediction is {class_labels[pred.argmax()]} ({pred.max()})")


            if SEG:
                
                in_clip = out_dict["input"]

                in_clip = (in_clip.detach()*255).to(torch.uint8)
                in_clip = in_clip.permute(0,1,3,4,2)
                if in_clip.shape[-1] == 1:
                    in_clip = in_clip.repeat(1,1,1,1,3)

                seg_pred = out_dict["pred_seg"]

                seg_pred = seg_pred.argmax(dim = 2, keepdim = True)

                seg_pred = colormap[seg_pred.squeeze(2).detach().cpu().numpy()]

                pred_lb = class_labels[pred.argmax()]

                gif_name = f"input{idx}_seg_pred.gif"
                
                path = "."
                gif_path = os.path.join(path, gif_name)


                gif_video = saveSegPredAsGIF(in_clip.squeeze(0), seg_pred.squeeze(0), pred_lb, gif_path, filename = gif_name)
            
