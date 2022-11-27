#Code Adopted from: https://github.com/mit-han-lab/temporal-shift-module 

# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

# import sys
# sys.path.append('code/models')

from model.video_models.transforms import *
from torch.nn.init import normal_, constant_

from model.video_models import space_time_aggregator as st

import torch.nn.functional as F

class TSN(nn.Module):
    def __init__(
                    self, num_class, num_channels, num_segments, modality,
                    base_model='resnet101', new_length=None,
                    st_consensus_type="SavgDropTavg", before_softmax=True,
                    dropout=0.8, img_feature_dim=256,
                    crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                    #  is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                    is_shift=True, shift_div=8, shift_place='block', fc_lr5=False,
                    #  temporal_pool=False, non_local=False
                    seq_train = False,
                    num_seq_class = -1,
                    return_features = False,
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
        
        self.seq_train = seq_train
        self.num_seq_class = num_seq_class

        self.return_features = return_features
        
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
        
        if self.seq_train:
            # self.seq_fc = nn.Sequential(nn.Linear(self.num_class, int(self.num_class/2)), nn.ReLU(), nn.Linear(int(self.num_class/2), self.num_seq_class))
            # self.seq_fc = nn.Sequential(nn.Sigmoid(), nn.Linear(self.num_class, int(self.num_class/2)), nn.ReLU(), nn.Linear(int(self.num_class/2), self.num_seq_class))
            # self.seq_fc = nn.Sequential(nn.ReLU(), nn.Linear(self.num_class, int(self.num_class/2)), nn.ReLU(), nn.Linear(int(self.num_class/2), self.num_seq_class))
            # self.seq_fc = nn.Sequential(nn.Linear(self.num_class, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, self.num_seq_class))
            # self.seq_fc = nn.Sequential(nn.Sigmoid(), nn.Linear(self.num_class, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, self.num_seq_class))
            # self.seq_fc = nn.Sequential(nn.ReLU(), nn.Linear(self.num_class, 32), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(32, 32), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(32, 16), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, self.num_seq_class))
            # self.seq_fc = nn.Sequential(nn.Linear(self.num_class, 32), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(32, 32), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(32, 16), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, self.num_seq_class))
            # self.seq_fc = nn.Sequential(nn.Linear(self.num_class, 64), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, self.num_seq_class))
            # self.seq_fc = nn.Sequential(nn.Linear(self.num_class, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p = 0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, self.num_seq_class))
            self.seq_fc = nn.Sequential(nn.Linear(self.num_class, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, self.num_seq_class))

        std = 0.001
        if hasattr(new_fc, 'weight'):
            normal_(new_fc.weight, 0, std)
            constant_(new_fc.bias, 0)

        if self.base_model_name == "resnet18":
            num_features = self.base_model.layer4[0].downsample[1].num_features
        elif self.base_model_name == "resnet34":
            num_features = self.base_model.layer4[0].downsample[1].num_features
        else: 
            raise ValueError(f"Undefined num_features for base model = {self.base_model_name}!")

        #Define space time aggregration layer 
        if self.st_consensus_type == "VideoTransformer":
            self.consensus = st.VideoTransformer(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "VideoTransformer2":
            self.consensus = st.VideoTransformer2(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "AttentionPooling":
            self.consensus = st.AttentionPooling(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "AttentionPoolingLnorm":
            self.consensus = st.AttentionPoolingLnorm(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "TmaxSavgDropCls":
            self.consensus = st.TmaxSavgDropCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "TmaxDropSavgCls":
            self.consensus = st.TmaxDropSavgCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "SavgDropTmaxCls":
            self.consensus = st.SavgDropTmaxCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features, return_features = self.return_features)
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
                from model.video_models.temporal_shift import make_temporal_shift
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

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]


    def forward(self, input, no_reshape=False):
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
        if self.return_features:
            output, output_ft = output

        if self.seq_train:
            output = self.seq_fc(output)


        if self.return_features:
            return {"pred_cls": output, "pred_ft": output_ft}

        return {"pred_cls": output}

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

        if self.seq_train:
            x = self.seq_fc(x)

        target_activations[0] = target_activations[0].view((-1, self.num_segments) + target_activations[0].size()[1:])
               
        return target_activations, x

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
