#Code Adopted from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py 


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


from model.unet import unetUp


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
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool = nn.Identity()
        self.fc = nn.Identity()
        self.up_layer3 = unetUp(512, 256, is_deconv = True)
        self.up_layer2 = unetUp(256, 128, is_deconv = True)
        self.up_layer1 = unetUp(128, 64, is_deconv = True)
        # self.seg_layer = nn.Conv2d(64, num_seg_classes, 1)
        
        # self.upsample_4 = Upsample(512, 256, 2, 2)
        # self.up_layer4 = self._make_layer(block, 512, 512, layers[3], outplanes = 256, stride=1, dilate=replace_stride_with_dilation[2])
        # self.upsample_3 = Upsample(256, 128, 2, 2)
        # self.up_layer3 = self._make_layer(block, 256, 256, layers[2], outplanes = 128, stride=1, dilate=replace_stride_with_dilation[1])
        # self.upsample_2 = Upsample(128, 64, 2, 2)
        # self.up_layer2 = self._make_layer(block, 128, 128, layers[1], outplanes = 64, stride=1, dilate=replace_stride_with_dilation[0])
        # # self.upsample_1 = Upsample(64, 32, 2, 2)
        # self.upsample_1 = Upsample(64, 32, 4, 4)
        # self.upsample_1 = Upsample(256, 32, 16, 16)
        # self.up_layer1 = self._make_layer(block, 32+1, 32+1, layers[0], outplanes = 16)

        # self.seg_layer = nn.Conv2d(16, num_seg_classes, 1)
        # self.seg_layer = nn.Conv2d(128, num_seg_classes, 1)

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

        # #Classification
        # x = self.avgpool(x4)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        s3 = self.up_layer3(x3, x4)
        s2 = self.up_layer2(x2, s3)
        s = self.up_layer1(x1, s2)
        # s = self.seg_layer(s1)

        #Segmentation
        # s4 = self.upsample_4(x4)
        # s4 = torch.cat([s4, x3], dim=1)
        # s4 = self.up_layer4(s4)

        # s3 = self.upsample_3(s4)
        # s3 = torch.cat([s3, x2], dim=1)
        # s3 = self.up_layer3(s3)

        # s2 = self.upsample_2(s3)
        # s2 = torch.cat([s2, x1], dim=1)
        # s2 = self.up_layer2(s2)

        # s1 = self.upsample_1(s2)
        # s1 = self.upsample_1(s4)
        # s1 = torch.cat([s1, x_in], dim=1)
        # s1 = self.up_layer1(s1)

        # s = self.seg_layer(s1)
        # s = self.seg_layer(s3)

        # return x, s
        return s

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class SegResNet_v0(nn.Module):
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