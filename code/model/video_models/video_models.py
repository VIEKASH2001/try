import torch
import torch.nn as nn
import torch.nn.functional as F



def shift(x, temporal_width, channel_temp_mix=8, temp_mix = 1):
        nt, c, h, w = x.size()
        n_batch = nt // temporal_width
        x = x.view(n_batch, temporal_width, c, h, w)

        fold = c // channel_temp_mix

        out = torch.zeros_like(x)
        out[:, :-temp_mix, :fold] = x[:, temp_mix:, :fold]  # shift left
        out[:, temp_mix:, fold: 2 * fold] = x[:, :-temp_mix, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetConv2ResTsm(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, temporal_width=8, channel_temp_mix=8, temp_mix = 1, kernel_size = 3):
        super(unetConv2ResTsm, self).__init__()

        self.temporal_width = temporal_width
        self.channel_temp_mix = channel_temp_mix
        self.temp_mix = temp_mix

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, 1, 1),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, 1, 1),
                                       nn.ReLU(),)
    def forward(self, inputs):
        
        outputs = self.conv1(inputs)

        #Temporal Shift input
        outputs_shift = shift(outputs.clone(), temporal_width = self.temporal_width, channel_temp_mix = self.channel_temp_mix, temp_mix = self.temp_mix)

        outputs_shift = self.conv2(outputs_shift)

        outputs = outputs + outputs_shift
        return outputs

  
class unetSumUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True, kernel_size = 2):
        super(unetSumUp, self).__init__()

        self.conv = unetConv2(out_size, out_size, False)
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size = kernel_size, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        x = inputs1 + outputs2
        y = self.conv(x) 

        return y


class tsmNet(nn.Module):
# class tsm_unetResSumEclassifier(nn.Module):    

    def __init__(self, in_channels=3, n_classes=5, feature_scale=1, is_deconv=True, is_batchnorm=True, temporal_width = 25, temp_mix = 1):
        super(tsmNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2ResTsm(self.in_channels, filters[0], self.is_batchnorm, temporal_width=temporal_width, channel_temp_mix=8, temp_mix = temp_mix)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2ResTsm(filters[0], filters[1], self.is_batchnorm, temporal_width=temporal_width, channel_temp_mix=8, temp_mix = temp_mix)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2ResTsm(filters[1], filters[2], self.is_batchnorm, temporal_width=temporal_width, channel_temp_mix=8, temp_mix = temp_mix)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2ResTsm(filters[2], filters[3], self.is_batchnorm, temporal_width=temporal_width, channel_temp_mix=8, temp_mix = temp_mix)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2ResTsm(filters[3], filters[4], self.is_batchnorm, temporal_width=temporal_width, channel_temp_mix=8, temp_mix = temp_mix)
        self.maxpool5 = nn.MaxPool2d(kernel_size=4)

        self.classifier = nn.Sequential(
                                nn.Flatten(),
                                # nn.Linear(in_features = 3072, out_features = 500),
                                nn.Linear(in_features = 2304, out_features = 500),
                                nn.ReLU(),
                                nn.Linear(in_features = 500, out_features = self.n_classes)
                            )

    def forward(self, inputs):
        
        # # Change dim from BxCxTxHxW -> BxTxCxHxW
        # inputs = inputs.permute(0, 2, 1, 3, 4)

        n, t, c, h, w = inputs.size()
        inputs = inputs.reshape(-1, c, h, w)

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        maxpool5 = self.maxpool5(center)


        final = self.classifier(maxpool5)

        final = final.view(n, t, self.n_classes)

        #Taking max across time axis #TODO - GRG : Need to see if taking Avg is better
        final = final.mean(dim=1)

        return final


