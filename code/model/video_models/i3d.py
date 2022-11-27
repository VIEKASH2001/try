import torch.nn as nn
import torch
import numpy as np

# from divatools import paths
# from runner.params.i3d_params import I3DParams
import os


# _PATHS = paths.get_default_paths()

# _CHECKPOINT_PATHS = {
#         'rgb': os.path.join(_PATHS.SAVED_MODELS_DIR, 'i3d_pretrained/rgb_scratch.pkl'),
#         'flow': os.path.join(_PATHS.SAVED_MODELS_DIR, 'i3d_pretrained/flow_scratch.pkl'),
#         'rgb_imagenet': os.path.join(_PATHS.SAVED_MODELS_DIR, 'i3d_pretrained/rgb_imagenet.pkl'),
#         'flow_imagenet': os.path.join(_PATHS.SAVED_MODELS_DIR, 'i3d_pretrained/flow_imagenet.pkl'),
#     }

_CHECKPOINT_PATHS = {}

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            BasicConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            BasicConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            BasicConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            BasicConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            BasicConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class I3D(nn.Module):

    def __init__(self, input_channel = 3, num_classes=400, dropout_keep_prob = 1, spatial_squeeze=True):
        super(I3D, self).__init__()
        self.features = nn.Sequential(
            BasicConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3), # (64, 32, 112, 112)
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (64, 32, 56, 56)
            BasicConv3d(64, 64, kernel_size=1, stride=1), # (64, 32, 56, 56)
            BasicConv3d(64, 192, kernel_size=3, stride=1, padding=1),  # (192, 32, 56, 56)
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (192, 32, 28, 28)
            Mixed_3b(), # (256, 32, 28, 28)
            Mixed_3c(), # (480, 32, 28, 28)
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), # (480, 16, 14, 14)
            Mixed_4b(),# (512, 16, 14, 14)
            Mixed_4c(),# (512, 16, 14, 14)
            Mixed_4d(),# (512, 16, 14, 14)
            Mixed_4e(),# (528, 16, 14, 14)
            Mixed_4f(),# (832, 16, 14, 14)
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # (832, 8, 7, 7)
            Mixed_5b(), # (832, 8, 7, 7)
            Mixed_5c(), # (1024, 8, 7, 7)
            nn.AdaptiveAvgPool3d(output_size=(8, 1, 1)), # (1024, 8, 1, 1)
            #nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1), # (1024, 8, 1, 1)
            nn.Dropout3d(dropout_keep_prob),
            nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),# (400, 8, 1, 1)
        )
        self.spatial_squeeze = spatial_squeeze

    def forward(self, x):
        """
        Args:
            x (torch.autograd.Variable): Video image data (BxCxDxHxW). Minimum supported size DxHxW is 13x17x17.
        Returns:
            Variable: Softmax results
            Variable: Pre-softmax network outputs
        """

        #GRG: NxTxCxHxW -> NxCxTxHxW 
        x = x.transpose(1,2)
        if len(x.shape) < 6:
            y = self.features(x)

            if self.spatial_squeeze:
                y = y.squeeze(3)
                y = y.squeeze(3)

            averaged_y = torch.mean(y, 2)
        else:
            for idx in range(x.shape[1]):
                if x.shape[0] == 1:
                    y = self.features(torch.unsqueeze(x[:, idx, :, :, :, :], dim=0))
                else:
                    y = self.features(x[:, idx, :, :, :, :])

                if self.spatial_squeeze:
                    y = y.squeeze(3)
                    y = y.squeeze(3)

                if idx == 0:
                    averaged_y = torch.mean(y, 2)
                else:
                    averaged_y += torch.mean(y, 2)

            averaged_y /= float(x.shape[1])
        
        # return averaged_y
        return {"pred_cls": averaged_y}

    def load_pretrained(self, model_path):
        state_dict = torch.load(model_path)
        self_num_classes = self.features[18].bias.shape[0]
        dict_num_classes = state_dict['features.18.bias'].shape[0]
        if self_num_classes != dict_num_classes:
            print(('num_classes({}) differ from loaded model num_classes({}). '
                  'Ignoring loaded final layer weights.'.format(self_num_classes, dict_num_classes)))
            state_dict['features.18.weight'] = self.features[18].weight.data.clone()
            state_dict['features.18.bias'] = self.features[18].bias.data.clone()
        self.load_state_dict(state_dict)


class I3DJoint(nn.Module):

    def __init__(self, rgb_model, flow_model):
        """
        Combines outputs of two I3D networks.

        Args:
            rgb_model (I3D): RGB model
            flow_model (I3D): Optical flow model
        """
        super(I3DJoint, self).__init__()
        self.rgb_model = rgb_model
        self.flow_model = flow_model

    def forward(self, x):
        """
        Args:
            x (2-tuple of torch.autograd.Variables): Contains rgb and optical flow data respectively.
                Absolute minimum supported size DxHxW is 13x17x17
        Returns:
            Variable: Softmax results
            Variable: Pre-softmax network outputs
        """

        rgb_y = self.rgb_model(x[0])
        flow_y = self.flow_model(x[1])

        y = rgb_y + flow_y
        return y


class pytorchI3D(nn.Module):

    def __init__(self, input_channel = 3, num_classes = 4, pretrained = True):
        super(pytorchI3D, self).__init__()

        #Torch Hub BugFix: https://stackoverflow.com/questions/68901236/urllib-error-httperror-http-error-403-rate-limit-exceeded-when-loading-resnet1
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

        # https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
        # https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained = pretrained)

        #Change model input channels
        self.model.blocks[0].conv = nn.Conv3d(input_channel, 64,  kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)

        #Change model output classes
        self.model.blocks[6].proj = nn.Linear(in_features=2048, out_features = num_classes, bias=True)
        

    def forward(self, input):

        #NxTxCxHxW -> NxCxTxHxW 
        input = input.transpose(1,2)
        
        output = self.model(input)

        return {"pred_cls": output}


def i3d(pretrained, model_type, **kwargs):
    """
    Initialize I3D network.

    Args:
        pretrained (boolean): If true load weights from pretrained model.
        model_type (int): Defines model type to use
            I3DParams.MODEL_TYPE_RGB: 3 channel input to I3D optionally loaded with pre-trained I3D RGB weights.
            I3DParams.MODEL_TYPE_OPT_FLOW: 2 channel input to I3D optionally pre-loaded with optical flow weights.
            I3DParams.MODEL_TYPE_JOINT: Creates combined model with both RGB and optical flow inputs.
        **kwargs: Extra arguments passed to I3D.__init__(...)
    Returns:
        I3D or I3DJoint model (depending on model_type)
    """
    rgb_model, flow_model = None, None
    if model_type in [I3DParams.MODEL_TYPE_RGB, I3DParams.MODEL_TYPE_JOINT]:
        kwargs['input_channel'] = 3
        rgb_model = I3D(**kwargs)
        if pretrained:
            rgb_model.load_pretrained(_CHECKPOINT_PATHS['rgb_imagenet'])
    if model_type in [I3DParams.MODEL_TYPE_OPT_FLOW, I3DParams.MODEL_TYPE_JOINT]:
        kwargs['input_channel'] = 2
        flow_model = I3D(**kwargs)
        if pretrained:
            flow_model.load_pretrained(_CHECKPOINT_PATHS['flow_imagenet'])

    if model_type == I3DParams.MODEL_TYPE_RGB:
        model = rgb_model
    elif model_type == I3DParams.MODEL_TYPE_OPT_FLOW:
        model = flow_model
    else:  # MODEL_TYPE_JOINT
        model = I3DJoint(rgb_model, flow_model)

    return model
