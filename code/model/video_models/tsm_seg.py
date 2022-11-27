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
    def __init__(self, num_class, num_seg_class, num_channels, num_segments, modality,
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
        elif self.st_consensus_type == "AttentionPooling":
            self.consensus = st.AttentionPooling(num_segments = self.num_segments, num_classes = self.num_class, num_features = num_features)
        elif self.st_consensus_type == "TmaxSavgDropCls":
            self.consensus = st.TmaxSavgDropCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "TmaxDropSavgCls":
            self.consensus = st.TmaxDropSavgCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "SavgDropTmaxCls":
            self.consensus = st.SavgDropTmaxCls(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        elif self.st_consensus_type == "SavgDropClsTavg":
            self.consensus = st.SavgDropClsTavg(new_fc, dprob = self.dropout, num_segments = self.num_segments, num_features = num_features)
        else:
            raise ValueError(f"Unsupported cfg.ST_CONCENSUS = {self.st_consensus_type}!")

        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            # self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            
            from model import resnet_seg
            # self.base_model = getattr(model.resnet_seg, base_model)(True if self.pretrain == 'imagenet' else False)
            self.base_model = getattr(resnet_seg, base_model)(True if self.pretrain == 'imagenet' else False, num_seg_classes = self.num_seg_class)
    
            
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


    def forward(self, input, no_reshape=False):
        if not no_reshape:
            base_out, seg_out = self.base_model(input.view((-1,) + input.size()[-3:]))
        else:
            base_out, seg_out = self.base_model(input)


        output = self.consensus(base_out)

        nt, c, h, w = seg_out.shape
        seg_out = seg_out.reshape(-1, self.num_segments, c, h, w)

        return {"pred_cls": output, "pred_seg": seg_out}

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
