"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-23 15:42:00
 * @modify date 2022-01-26 20:48:20
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
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
import torch
import torchvision.models.video as video_models
from torch import nn
from torch.nn import functional as F
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout = 0.2, activation='relu'):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def attention(self, q, k, v, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_model)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output, scores

    def forward(self, q, k, v):
        bs, n, c, f = q.size(0), q.size(1), q.size(2), q.size(3)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, n, c, self.d_model)
        q = self.q_linear(q).view(bs, n, c, self.d_model)
        v = self.v_linear(v).view(bs, n, c, self.d_model)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores, attention = self.attention(q, k, v, None, self.dropout)
        output = self.activation(self.out(scores)).transpose(1,2)
        return output, (attention)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward=64, dropout=0.2, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.self_attn = MultiHeadAttention(hidden_dim, dropout=dropout, activation=activation)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        src2, weights = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights

class VideoTransformer(nn.Module):
    def __init__(self, num_segments, num_classes, num_features, hidden_dim = 128, num_layers = 1, dim=1):
        super(VideoTransformer, self).__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.num_clips = num_segments
        self.feature_dim = num_features
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionalEncoding(self.feature_dim, max_len=self.num_clips)
        self.feature_expansion = nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, self.hidden_dim), nn.ReLU()) for i in range(self.num_classes)])
        # self.initial_classifiers = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for i in range(self.num_classes)])
        self.encoder_layers1 = nn.ModuleList([TransformerEncoderLayer(hidden_dim=self.hidden_dim, dim_feedforward=self.hidden_dim) for i in range(self.num_layers)])
        self.encoder_layers2 = nn.ModuleList([TransformerEncoderLayer(hidden_dim=self.hidden_dim, dim_feedforward=self.hidden_dim) for i in range(self.num_layers)])
        self.final_classifiers = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for i in range(self.num_classes)])
        self.alpha = nn.Parameter(torch.tensor([0.0]*self.num_layers)) #torch.tensor([0.0]*self.num_layers)
        self.sigmoid = nn.Sigmoid()

        self.module = nn.Sequential(
                        # nn.Dropout(p=self.dprob),
                        nn.AdaptiveAvgPool2d(1), 
                        nn.Flatten(),
                        # fc
                    )

    def forward(self, features):

        nt, chw = features.size()

        h = int((chw/self.feature_dim)**0.5)

        # # NTxCHW -> NxTxCxHxW
        # features = features.reshape(-1, self.num_clips, self.feature_dim, h, h)

        # NTxCHW -> NTxCxHxW
        input_tensor = features.reshape(nt, self.feature_dim, h, h)

        # NTxCxHxW -> NTxC
        output = self.module(input_tensor)

        # NTxC -> NxTxC
        features = output.view((-1, self.num_clips) + output.size()[1:]) 

        # activations = {}
        features = self.positional_encoding(features)
        features = [self.feature_expansion[i](features) for i in range(self.num_classes)]
        
        # NxTxCxF
        expanded_features = torch.stack(features, dim=0).permute(1, 2, 0, 3)
        # initial_outputs = [self.initial_classifiers[i](expanded_features[:,:,i,:]) for i in range(self.num_classes)]
        # initial_outputs = torch.stack(initial_outputs, dim=2).squeeze(-1)
        # activations['init_output'] = initial_outputs
        features = expanded_features
        for i in range(self.num_layers):
            encoder_output1 = self.encoder_layers1[i](features)[0]
            encoder_output2 = self.encoder_layers2[i](features.transpose(1,2))[0].transpose(1,2)
            features = (self.sigmoid(self.alpha[i]) * encoder_output1) + ((1 - self.sigmoid(self.alpha[i])) *  encoder_output2)
        final_outputs = [self.final_classifiers[i](features[:,:,i,:]) for i in range(self.num_classes)]
        final_outputs = torch.stack(final_outputs, dim=2).squeeze(-1)
        
        # activations['final_output'] = final_outputs
        # return activations


        # NxTxC -> NxC
        # output = F.adaptive_max_pool1d(output.permute(0,2,1), 1).squeeze()
        final_outputs = final_outputs.max(dim=self.dim, keepdim=False)[0]

        return final_outputs


class VideoTransformer2(nn.Module):
    def __init__(self, num_segments, num_classes, num_features, num_layers = 1, dim=1):
        super(VideoTransformer2, self).__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.num_clips = num_segments
        self.feature_dim = num_features
        self.num_layers = num_layers
        self.hidden_dim = num_features
        self.positional_encoding = PositionalEncoding(self.feature_dim, max_len=self.num_clips)
        
        self.encoder_layers1 = nn.ModuleList([TransformerEncoderLayer(hidden_dim=self.hidden_dim, dim_feedforward=self.hidden_dim) for i in range(self.num_layers)])
        self.encoder_layers2 = nn.ModuleList([TransformerEncoderLayer(hidden_dim=self.hidden_dim, dim_feedforward=self.hidden_dim) for i in range(self.num_layers)])
        self.final_classifiers = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for i in range(self.num_classes)])
        self.alpha = nn.Parameter(torch.tensor([0.0]*self.num_layers)) #torch.tensor([0.0]*self.num_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):

        nt, chw = features.size()

        h = int((chw/self.feature_dim)**0.5)

        # NTxCHW -> NHWxTxC
        features = features.reshape(-1, self.num_clips, self.feature_dim)

        features = self.positional_encoding(features)

        # NHWxTxC -> NxTxHWxC
        features = features.reshape(-1, self.num_clips, h*h, self.feature_dim)
        
        for i in range(self.num_layers):
            encoder_output1 = self.encoder_layers1[i](features)[0]
            encoder_output2 = self.encoder_layers2[i](features.transpose(1,2))[0].transpose(1,2)
            features = (self.sigmoid(self.alpha[i]) * encoder_output1) + ((1 - self.sigmoid(self.alpha[i])) *  encoder_output2)
        final_outputs = [self.final_classifiers[i](features[:,:,i,:]) for i in range(self.num_classes)]
        final_outputs = torch.stack(final_outputs, dim=2).squeeze(-1)
        
        # NxTxC -> NxC
        final_outputs = final_outputs.max(dim=self.dim, keepdim=False)[0]

        return final_outputs


class AttentionPooling(nn.Module):
    def __init__(self, num_segments, num_classes, num_features, num_layers = 3, dim=1, height=7, dropout = 0.2):
        super(AttentionPooling, self).__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.num_clips = num_segments
        self.feature_dim = num_features
        self.num_layers = num_layers
        self.hidden_dim = num_features
        self.positional_encoding = PositionalEncoding(self.feature_dim, max_len=self.num_clips*height*height**2)
        
        # self.attention_pooling_layers = nn.ModuleList([nn.Linear(self.hidden_dim, int(self.hidden_dim/4), bias = False) for i in range(self.num_layers)])
        self.attention_pooling_layers = nn.ModuleList([nn.Linear(self.hidden_dim, 1, bias = False) for i in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.attention_encoder_layer = nn.Sequential(nn.Linear(self.hidden_dim, int(self.hidden_dim/4)), nn.ReLU())

        # self.final_classifiers = nn.Linear(int(self.num_layers*self.hidden_dim/4), self.num_classes if self.num_classes > 2 else 1)
        self.final_classifiers = nn.Linear(int(self.num_layers*self.hidden_dim/4), self.num_classes)
        

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

        # final_features = torch.column_stack(attended_features)
        final_features = torch.cat(attended_features, dim = -1)
        
        # NxF -> NxC
        final_outputs = self.final_classifiers(final_features)

        return final_outputs


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
                clip_input_sz=(224,224),
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

        self.clip_input_sz = clip_input_sz

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

        # #Conver to rect image - 1920x1080p
        # left_upper_point, left_lower_point, right_upper_point, right_lower_point = [[795.1225906658183, 204.40587222575687], [411.12618361465627, 762.2111793106028], [1113.0985628204648, 203.0585164115423], [1490.358190800554, 766.2532467532467]]
        # #Conver to rect image - 1080x720p
        # left_upper_point, left_lower_point, right_upper_point, right_lower_point = [[448.0625295475402, 140.2469618176258], [233.15110264467864, 501.99228287772195], [624.9306293278455, 140.2469618176258], [839.8420562307072, 506.6642704190885]]
        #Conver to rect image - 1280x720p
        left_upper_point, left_lower_point, right_upper_point, right_lower_point = [[532.7117133401929, 141.58181540087344], [276.419825356656, 505.9968436274647], [740.2814455351927, 139.57953502600208], [993.2361995606104, 497.9877221279792]]
        
        #Conver to rect image - 1080x720p
        # 'Image_262499828648_clean':[[448.0625295475402, 140.2469618176258], [233.15110264467864, 501.99228287772195], [624.9306293278455, 140.2469618176258], [839.8420562307072, 506.6642704190885]]
        # special variables
        # function variables
        # 0:[448.0625295475402, 140.2469618176258]
        # 1:[233.15110264467864, 501.99228287772195]
        # 2:[624.9306293278455, 140.2469618176258]
        # 3:[839.8420562307072, 506.6642704190885]

        #Conver to rect image - 1280x720p
        # "Image_262499828648_clean" : [[532.7117133401929, 141.58181540087344], [276.419825356656, 505.9968436274647], [740.2814455351927, 139.57953502600208], [993.2361995606104, 497.9877221279792]]
        # special variables
        # function variables
        # 0:[532.7117133401929, 141.58181540087344]
        # 1:[276.419825356656, 505.9968436274647]
        # 2:[740.2814455351927, 139.57953502600208]
        # 3:[993.2361995606104, 497.9877221279792]

        # resize_x, resize_y = video_frames.shape[-1]/1920.0, video_frames.shape[-2]/1080.0
        
        # def resizeCoordinates(pt):
        #     x,y = pt
        #     return [x*resize_x, y*resize_y]

        # left_upper_point = resizeCoordinates(left_upper_point)
        # left_lower_point = resizeCoordinates(left_lower_point)
        # right_upper_point = resizeCoordinates(right_upper_point)
        # right_lower_point = resizeCoordinates(right_lower_point)

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
        print('preprocess_input():  calculated Polar -> Rect ends with width, height of ', width, height)

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

        # Resize frames:  This controls the model's expected input resolution and
        #                 the image-rectification interpolation output resolution
        #                 Note:  TSM->self.input_size is ignored.
        # image_size = (224, 224)
        # image_size = (416, 416)
        image_size = self.clip_input_sz
        print(f"Using clip_input_sz = {image_size}")

        # video_frames = [cv2.resize(f, image_size) for f in video_frames]
        
        n, t, c, h, w = video_frames.shape
        video_frames = video_frames.reshape(-1, c, h, w)
        # video_frames = F.interpolate(video_frames, size = image_size, mode = "bicubic", align_corners = True)
        # video_frames = F.interpolate(video_frames, size = (1, 224, 224), mode = "trilinear", align_corners = True)
        video_frames = F.interpolate(video_frames, size = image_size, mode = "bilinear", align_corners = True)
        
        video_frames = video_frames.reshape(n, t, c, image_size[0], image_size[1])

        # # clip = (clip - clip.min())/(clip.max() - clip.min())
        # video_frames = video_frames.clamp(min=0.0, max=255.0) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)
        video_frames = video_frames.clamp(min=0.0, max=1.0) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)

        # video_frames = video_frames/255.0

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

        output = F.softmax(output, dim=-1)

        return output

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




import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm,
                          target_names,
                          path, 
                          showFig = False,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          prefix = ''):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

   

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            # plt.text(j, i, "{:0.4f}".format(cm[i, j]),
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
    plt.savefig(os.path.join(path, f"{prefix}_confusion_matrix.png"))

    if showFig:
        plt.show()
    
    return fig


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, multilabel_confusion_matrix



import os

from statistics import mode

if __name__ == "__main__":

    # CHANGE
    JOHN_SYS = False
    Savg = True

    if JOHN_SYS:
        #model_path = "/Users/jgaleotti/Research/POCUS_AI_DARPA_src/PTX/v1/tsm_DARPA-Dataset_Rct_SavgDropTmaxCls_T7_A_epoch=9-step=79.ckpt"
        # model_path = "tsm_DARPA-Dataset_Rct_SavgDropTmaxCls_T7_A_epoch=9-step=79.ckpt" # Uses 224x224 input, TSN.st_consensus_type = "SavgDropTmaxCls"
        # model_path = "tsm_DARPA-Dataset_AttentionPooling_RandTS_large_B_epoch=16-step=135.ckpt" # Uses 416x416 input, TSN.st_consensus_type = "AttentionPooling"

        if Savg:
            model_path = "tsm_DARPA-Dataset_Rct_SavgDropTmaxCls_T7_A_epoch=9-step=79.ckpt" # Uses 224x224 input, TSN.st_consensus_type = "SavgDropTmaxCls"
            st_consensus_type = "SavgDropTmaxCls"
            clip_input_sz = (224, 224)
        else:
            model_path = "tsm_DARPA-Dataset_AttentionPooling_RandTS_large_B_epoch=16-step=135.ckpt" # Uses 416x416 input, TSN.st_consensus_type = "AttentionPooling"
            st_consensus_type = "AttentionPooling"
            clip_input_sz = (416, 416)

        # video_path = "image_1394469579519_clean.mp4"

        video_root_path = "."

    else:
        if Savg:
            # model_path = "/home/grg/Research/DARPA-Pneumothorax/results_v0/Pneumothorax_Exps/tsm_DARPA-Dataset_Rct_SavgDropTmaxCls_T7_A/Pneumothorax_Exps_Pneumothorax_Exps/1dn54sdi_0/checkpoints/epoch=9-step=79.ckpt"
            model_path = "/data2/exp_results/DARPA-Pneumothorax/results_v0/Pneumothorax_Exps/tsm_DARPA-Dataset_Rct_SavgDropTmaxCls_T7_A/Pneumothorax_Exps_Pneumothorax_Exps/1dn54sdi_0/checkpoints/epoch=9-step=79.ckpt"
            st_consensus_type = "SavgDropTmaxCls"
            clip_input_sz = (224, 224)
        else:
            # model_path = "/home/grg/Research/DARPA-Pneumothorax/results_v0/Pneumothorax_Exps/tsm_DARPA-Dataset_AttentionPooling_RandTS_large_B/Pneumothorax_Exps_Pneumothorax_Exps/54tlyq6m_0/checkpoints/epoch=16-step=135.ckpt"
            model_path = "/data2/exp_results/DARPA-Pneumothorax/results_v0/Pneumothorax_Exps/tsm_DARPA-Dataset_AttentionPooling_RandTS_large_B/Pneumothorax_Exps_Pneumothorax_Exps/54tlyq6m_0/checkpoints/epoch=16-step=135.ckpt"
            st_consensus_type = "AttentionPooling"
            clip_input_sz = (416, 416)

        # video_root_path = "/data1/datasets/DARPA-Dataset"
        # video_root_path = "/data1/datasets/DARPA POCUS AI/Test dataset - Phase 1 - Unlabeled sliding vs no-sliding/"
        video_root_path = "/data1/datasets/DARPA_POCUS_AI/Test dataset-Phase1/"

        # video_path = "/data1/datasets/DARPA-Dataset/NoSliding/image_1394469579519_clean.mp4" #fold-B difficulty=1
        # video_path = "/data1/datasets/DARPA-Dataset/NoSliding/image_1083297968960_clean.mp4" #fold-test difficulty=0
        # video_path = "/data1/datasets/DARPA-Dataset/NoSliding/image_1087766719219_clean.mp4" #fold-test difficulty=-1

        # video_path = "/data1/datasets/DARPA-Dataset/Sliding/image_10891015221417_clean.mp4" #fold-test difficulty=-1
        # video_path = "/data1/datasets/DARPA-Dataset/Sliding/image_677741729740_clean.mp4" #fold-test difficulty=-1
    

    # testsuit_videos = [
    #     ["NoSliding/image_4641643404894_CLEAN.mp4", "fold-test difficulty=-2"],
    #     ["NoSliding/image_1087766719219_clean.mp4", "fold-test difficulty=-1"],
    #     ["NoSliding/image_634125159704_CLEAN.mp4", "fold-test difficulty=0"],
    #     ["NoSliding/image_642169070951_clean.mp4", "fold-test difficulty=0"],
    #     ["NoSliding/image_1083297968960_clean.mp4", "fold-test difficulty=0"],
    #     ["NoSliding/image_1895283541879_clean.mov", "fold-test difficulty=1"],
    #     ["NoSliding/image_1884162273498_clean.mov", "fold-test difficulty=1"],
    #     ["NoSliding/image_603665940081_clean.mov", "fold-test difficulty=2"],
    #     ["NoSliding/image_614587120545_clean.mov", "fold-test difficulty=2"],

    #     ["Sliding/image_104543812690743_CLEAN.mov", "fold-test difficulty=-2"],
    #     ["Sliding/image_104548309385533_CLEAN.mov", "fold-test difficulty=-2"],
    #     ["Sliding/image_104932526155699_CLEAN.mov", "fold-test difficulty=-2"],
    #     ["Sliding/image_10891015221417_clean.mp4", "fold-test difficulty=-1"],
    #     ["Sliding/image_1180496934444_clean.mp4", "fold-test difficulty=-1"],
    #     ["Sliding/image_677741729740_clean.mp4", "fold-test difficulty=-1"],
    #     ["Sliding/image_73815992352100_clean.mp4", "fold-test difficulty=0"],
    #     ["Sliding/image_74132233134844_clean.mp4", "fold-test difficulty=0"],
    #     ["Sliding/image_588413346180_CLEAN.mp4", "fold-test difficulty=2"],
        
    # ]
    
    testsuit_videos = [
            ["image_2673074007832_CLEAN.mp4",	"Sliding", "yes"],
            ["image_2518177532892_CLEAN.mp4",	"Sliding", "yes"],
            ["image_307870966391_CLEAN.mp4",	"Sliding", "yes"],
            ["image_527803396828_CLEAN.mp4",	"Sliding", "yes"],
            ["image_2100361349561_CLEAN.mp4",	"Sliding", "no"],
            ["image_1991804868665_CLEAN.mp4",	"NoSliding", "no"],
            ["image_432792096448_CLEAN.mp4",	"Sliding", "yes"],
            ["image_2375004178780_CLEAN.mp4",	"NoSliding", "yes"],
            ["image_225130637841202_CLEAN.mp4",	"Sliding", "no"],
            ["image_703682641552_CLEAN.mp4",	"Sliding", "yes"],
            ["image_1036584980280_CLEAN.mp4",	"Sliding", "no"],
            ["image_931984558757_CLEAN.mp4",	"NoSliding", "no"],
            ["image_987146487434_CLEAN.mp4",	"Sliding", "yes"],
            ["image_6692463753591_CLEAN.mp4",	"NoSliding", "yes"],
            ["image_2475978478013_CLEAN.mp4",	"NoSliding", "no"],
            ["image_1355315203127_CLEAN.mp4",	"Sliding", "yes"],
            ["image_1604910716625_CLEAN.mp4",	"NoSliding", "yes"],
            ["image_740584633467_CLEAN.mp4",	"Sliding", "yes"],
            ["image_2370853937896_CLEAN.mp4",	"Sliding", "yes"],
            ["image_4065361175427_CLEAN.mp4",	"Sliding", "no"],
            ["image_1675609772328_CLEAN.mp4",	"Sliding", "yes"],
            ["image_1710325448148_CLEAN.mp4",	"Sliding", "no"],
            ["image_1447186382728_CLEAN.mp4",	"Sliding", "no"],
            ["image_1215182877868_CLEAN.mp4",	"Sliding", "yes"],
            ["image_4217982672400_CLEAN.mp4",	"NoSliding", "yes"],
            ["image_4227031119375_CLEAN.mp4",	"NoSliding", "yes"],
            ["image_383588065321_CLEAN.mp4",	"Sliding", "yes"],
            ["image_9970708112236_CLEAN.mp4",	"Sliding", "yes"],
            ["image_270509486562_CLEAN.mp4",	"NoSliding", "no"],
            ["image_706555992698_CLEAN.mp4",	"NoSliding", "no"],
            ["image_3012822351141_CLEAN.mp4",	"Sliding", "yes"],
            ["image_222706744966_CLEAN.mp4",	"Sliding", "yes"],
            ["image_358997745435_CLEAN.mp4",	"Sliding", "no"],
            ["image_2719958257554_CLEAN.mp4",	"Sliding", "yes"],
            ["image_560395551296_CLEAN.mp4",	"NoSliding", "yes"],
            ["image_595115042480_CLEAN.mp4",	"NoSliding", "no"],
            ["image_620422280751_CLEAN.mp4",	"NoSliding", "no"],
            ["image_3512795900742_CLEAN.mp4",	"NoSliding", "no"],
            ["image_3945589816827_CLEAN.mp4",	"NoSliding", "yes"],
            ["image_456885889303_CLEAN.mp4",	"NoSliding", "no"],
            ["image_9367839490019_CLEAN.mp4",	"NoSliding", "no"],
            ["image_9401889534225_CLEAN.mp4",	"NoSliding", "no"],
            ["image_233507706680_CLEAN.mp4",	"NoSliding", "no"],
            ["image_507659607045_CLEAN.mp4",	"NoSliding", "yes"],
            ["image_543109924479_CLEAN.mp4",	"NoSliding", "yes"],
        ]

    # testsuit_videos = [
    #     ['image_1036584980280_CLEAN.mp4', 'video-0', 'sliding'], 
    #     ['image_1215182877868_CLEAN.mp4', 'video-1', 'sliding'], 
    #     ['image_1355315203127_CLEAN.mp4', 'video-2', 'sliding'], 
    #     ['image_1447186382728_CLEAN.mp4', 'video-3', ''], 
    #     ['image_1604910716625_CLEAN.mp4', 'video-4'], 
    #     ['image_1675609772328_CLEAN.mp4', 'video-5'], 
    #     ['image_1710325448148_CLEAN.mp4', 'video-6'], 
    #     ['image_1991804868665_CLEAN.mp4', 'video-7'], 
    #     ['image_2100361349561_CLEAN.mp4', 'video-8'], 
    #     ['image_222706744966_CLEAN.mp4', 'video-9'], 
    #     ['image_225130637841202_CLEAN.mp4', 'video-10'], 
    #     ['image_233507706680_CLEAN.mp4', 'video-11'], 
    #     ['image_2370853937896_CLEAN.mp4', 'video-12'], 
    #     ['image_2375004178780_CLEAN.mp4', 'video-13'], 
    #     ['image_2475978478013_CLEAN.mp4', 'video-14'], 
    #     ['image_2518177532892_CLEAN.mp4', 'video-15'], 
    #     ['image_2673074007832_CLEAN.mp4', 'video-16'], 
    #     ['image_270509486562_CLEAN.mp4', 'video-17'], 
    #     ['image_2719958257554_CLEAN.mp4', 'video-18'], 
    #     ['image_3012822351141_CLEAN.mp4', 'video-19'], 
    #     ['image_307870966391_CLEAN.mp4', 'video-20'], 
    #     ['image_3512795900742_CLEAN.mp4', 'video-21'], 
    #     ['image_358997745435_CLEAN.mp4', 'video-22'], 
    #     ['image_383588065321_CLEAN.mp4', 'video-23'], 
    #     ['image_3945589816827_CLEAN.mp4', 'video-24'], 
    #     ['image_4065361175427_CLEAN.mp4', 'video-25'], 
    #     ['image_4217982672400_CLEAN.mp4', 'video-26'], 
    #     ['image_4227031119375_CLEAN.mp4', 'video-27'], 
    #     ['image_432792096448_CLEAN.mp4', 'video-28'], 
    #     ['image_456885889303_CLEAN.mp4', 'video-29'], 
    #     ['image_507659607045_CLEAN.mp4', 'video-30'], 
    #     ['image_527803396828_CLEAN.mp4', 'video-31'], 
    #     ['image_543109924479_CLEAN.mp4', 'video-32'], 
    #     ['image_560395551296_CLEAN.mp4', 'video-33'], 
    #     ['image_595115042480_CLEAN.mp4', 'video-34'],
    #     ['image_620422280751_CLEAN.mp4', 'video-35'],
    #     ['image_6692463753591_CLEAN.mp4', 'video-36'], 
    #     ['image_703682641552_CLEAN.mp4', 'video-37'], 
    #     ['image_706555992698_CLEAN.mp4', 'video-38'],
    #     ['image_740584633467_CLEAN.mp4', 'video-39'], 
    #     ['image_931984558757_CLEAN.mp4', 'video-40'], 
    #     ['image_9367839490019_CLEAN.mp4', 'video-41'], 
    #     ['image_9401889534225_CLEAN.mp4', 'video-42'], 
    #     ['image_987146487434_CLEAN.mp4', 'video-43'], 
    #     ['image_9970708112236_CLEAN.mp4', 'video-44']
    # ]

    
    f = open(f"Test_suit_results_for_{st_consensus_type}.txt", 'w')
    f.write(f'\n')
    f.write(f'Model: {st_consensus_type} \n')

    # Initialize model
    # Input dimensions are NxTxCxHxW = 1x15x1x224x224 (with rectification TxHxW = 15x1080x1920)
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
                    # st_consensus_type = "SavgDropTmaxCls"
                    # st_consensus_type = "AttentionPooling",
                    st_consensus_type = st_consensus_type,
                    clip_input_sz = clip_input_sz,
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
    
    video_pred_labels = []

    for video_name, video_desc, clinician_agreement in testsuit_videos:
        
        video_path = os.path.join(video_root_path, video_name)

        if not os.path.exists(video_path):
            raise Exception(f"File doesn't exists! {video_path}")

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
        video_frames = [cv2.resize(f, (1280, 720))/255.0 for f in video_frames]
        
        
        #Model Forward pass
        print('Evaluating model')
        # class_labels = [ 'no-sliding', 'sliding', ]
        class_labels = [ 'NoSliding', 'Sliding', ]
        
        idx_count = int(len(video_frames)/model.num_segments)
        print('idx_count = ', idx_count, ', frame count is ', len(video_frames), ', segment count is ', model.num_segments)
        
        print(f"\n\n *** {video_name} {video_desc} *** \n\n")
        f.write(f"\n\n *** {video_name} {video_desc} *** \n\n")

  

        clip_pred_label = []
        with torch.no_grad():

            # temp_sampling_method = "equi_temp_sampling"
            temp_sampling_method = "rand_temp_sampling"
            clip_width = model.num_segments
            # no_multi_clip = 3
            no_multi_clip = idx_count


                
            for idx in range(idx_count):

                if temp_sampling_method == "equi_temp_sampling":
                    clip_stride = np.floor(len(video_frames)/clip_width)

                    start_frame = np.arange(0, clip_stride, int(clip_stride/no_multi_clip))[idx]
                    
                    clip_frame_ids = np.arange(start_frame, len(video_frames), clip_stride).astype(np.int)

                    #Slice the clips
                    clip_frame_ids = clip_frame_ids[:clip_width]

                    # idx_count = no_multi_clip

                elif temp_sampling_method == "rand_temp_sampling":

                    clip_stride = 1
                    start_frame = np.arange(0, len(video_frames), clip_width)[idx]
                    
                    clip_frame_ids = np.arange(start_frame, start_frame+(clip_width*clip_stride), clip_stride).astype(np.int)

                    # idx_count = int(len(video_frames)/model.num_segments)
                    
                # print('    preparing frame ', idx)
                # clip = video_frames[idx*model.num_segments:(idx+1)*(model.num_segments)]

                clip = np.array(video_frames)[clip_frame_ids]
                
                # Remap clip to have dimensions NxTxCxHxW:
                clip = torch.tensor(clip).unsqueeze(1).unsqueeze(0).float()
                
                # print('    evaluating frame ', idx)
                pred = model(clip)
                
                # print(f"[{idx}] Model  pre-softmax outputs are {class_labels[0]}: {pred[0,0]},  {class_labels[1]}: {pred[0,1]}")
                
                # pred = F.softmax(pred)
                
                # print(f"[{idx}] Model post-softmax outputs are {class_labels[0]}: {pred[0,0]},  {class_labels[1]}: {pred[0,1]}")
                
                clip_pred_label.append(class_labels[pred.argmax()])
                
                print(f"    [clip-{idx}] Model prediction is {class_labels[pred.argmax()]} ({pred.max()})")
                f.write(f"  [clip-{idx}] Model prediction is {class_labels[pred.argmax()]} ({pred.max()}) \n")
        
        video_pred_labels.append(mode(clip_pred_label))

        
        print(f"\n Model prediction : {mode(clip_pred_label)}, target : {video_desc} [clinician_agreement : {clinician_agreement}] \n")
        f.write(f"\n Model prediction : {mode(clip_pred_label)}, target : {video_desc} [clinician_agreement : {clinician_agreement}] \n")

        # print(f"\n clinician_agreement : {clinician_agreement} \n")
        # f.write(f"\n clinician_agreement : {clinician_agreement} \n")

    # aa = [v.replace("no-sliding", "NoSliding").replace("sliding", "Sliding") for v in video_pred_labels]
    # acc = (np.array(testsuit_videos)[:,1] == aa).mean()
    acc = (np.array(testsuit_videos)[:,1] == video_pred_labels).mean()

    print(f"Accuracy of the {st_consensus_type} model is {acc}")
    f.write(f"\n\nAccuracy of the {st_consensus_type} model is {acc}\n\n")



    targets = np.array(testsuit_videos)[:,1]
    preds = np.array(video_pred_labels)

    targets[targets == class_labels[0]] = 0
    targets[targets == class_labels[1]] = 1

    preds[preds == class_labels[0]] = 0
    preds[preds == class_labels[1]] = 1


    targets = targets.astype(np.int)
    preds = preds.astype(np.int)

    accuracy = accuracy_score(targets, preds)

    confusionMatrix = confusion_matrix(targets, preds, labels = [0, 1])
        
    classificationReport = classification_report(targets, preds, labels = [0, 1], target_names = class_labels, digits=5)

    print(f"Testset : accuracy = {accuracy}")
    print(f"Testset : confusionMatrix = \n {confusionMatrix}")
    print(f"Testset : classificationReport = \n {classificationReport}")

    f.write(f"\n\nTestset : accuracy = {accuracy}\n\n")
    f.write(f"\n\nATestset : confusionMatrix = \n {confusionMatrix}\n\n")
    f.write(f"\n\nTestset : classificationReport = \n {classificationReport}\n\n")

    cm_fig = plot_confusion_matrix(confusionMatrix, target_names = class_labels, path = "./", normalize=False, prefix = "DARPA_Testset")
    norm_cm_fig = plot_confusion_matrix(confusionMatrix, target_names = class_labels, path = "./", normalize=True, prefix = f"DARPA_Testset_normalized")


    f.close()


    if JOHN_SYS:
        ########################################
        # Convert the model using CoreML Tools #
        ########################################
        
        import coremltools as ct
        
        print('CoreML Tools imported; setting parameters next')
        
        # Set the model in evaluation mode
        model.eval()
        
        # Specify the dimensions of the input, NxTxCxHxW:
        inference_batch_size = 1
        input_size = (inference_batch_size, model.num_segments,
                      model.num_channels, 720, 1280)
        #              model.num_channels, model.input_size, model.input_size)
        #              model.num_channels, model.input_size[0], model.input_size[1])
        
        # Specify the expected range of input pixel values
        input_pixel_min_value = 0.0
        input_pixel_max_value = 1.0
        
        # Setup values for conversion
        dummy_input = torch.rand(input_size)
        input_names = [ "input_images" ]
        output_names = [ "classLabel", "var_870"] #"var_809"] #"var_736"]
        input_descriptions = [ "Array of 15 grayscale frames, each of size 1280x720 with pixel values rescaled to between 0 to 1." ]
        output_descriptions = [ "Diagnosis output of either sliding or no-sliding", "Softmax classification probabilities for given input sample" ]
        
        # # Optional:  Print out a summary of the model:
        # from torchinfo import summary
        # print(summary(model, input_size))
        
        # Next, convert the model to TorchScript:  https://coremltools.readme.io/docs/pytorch-conversion
        #        Example:  https://coremltools.readme.io/docs/pytorch-conversion-examples
        
    #    # As a temporary test, convert the model to ONNX, then from ONNX to CoreML:
    #    print('Exporting model to ONNX...')
    #    torch.onnx.export(model,
    #                  dummy_input,
    #                  "TSMNet.onnx",
    #                  verbose=False,
    #                  input_names=input_names,
    #                  output_names=output_names,
    #                  export_params=True,
    #                  opset_version=11 # OpSet 11 is the smallest version that will successfully convert this model.
    #                  )
    #    # Convert from ONNX to Core ML (latest supported ios target is 13 for ONNX converter)
    #    #print('Exporting ONNX model to CoreML...')
    #    #mlmodel  = ct.converters.onnx.convert(model='TSMNet.onnx', minimum_ios_deployment_target='13')
    #    #print('Done Exporting ONNX model to CoreML')
        
        
        # Trace the model with random data; does not work if model contains conditionals, loops, etc.
        # See more details on scripting vs tracing here:  https://coremltools.readme.io/docs/model-scripting
        print('Tracing model...')
        traced_model = torch.jit.trace(model, dummy_input)
        # Optional, no need to actually save the intermediate traced model to disk:
        traced_model.save("TSMNet.pt")
        #out = traced_model(dummy_input)
        print('Converting to CoreML...')
        
        # To learn how to get better performance and more convenience when using images as inputs, see:
        # https://coremltools.readme.io/docs/image-inputs
        # Define the input type as image,
        # set pre-processing parameters to normalize the image
        # to have its values in the interval [-1,1]
        # as expected by the mobilenet model
        #image_input = ct.ImageType(shape=(model.num_channels, model.input_size, model.input_size, 3,), bias=[-1,-1,-1], scale=1/127)
            
        # set output class labels
        class_labels = [ 'no-sliding', 'sliding', ]
        classifier_config = ct.ClassifierConfig(class_labels)
        
        # ImageType is better and faster for inputs, but only works for rank 3 or 4.
        # TSM model is currently rank 5 input (NxTxCxHxW) so stick with Tensor for now.
        #image_input = ct.ImageType(shape=input_size)
        
        #mlmodel = ct.convert( traced_model, inputs=[image_input], classifier_config=classifier_config, )
        
        mlmodel = ct.convert( traced_model, inputs=[ct.TensorType( name=input_names[0], shape=input_size )], classifier_config=classifier_config, )
        
        # Set feature descriptions (these show up as comments in XCode)
        mlmodel.input_description[input_names[0]] = input_descriptions[0]
        mlmodel.output_description[output_names[0]] = output_descriptions[0]
        mlmodel.output_description[output_names[1]] = output_descriptions[1]
        mlmodel.author = "CMU  (Gautam Gare, John Galeotti, et. al)" #'"Original Paper: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen'
        mlmodel.license = "Proprietary and confidential - do not distribute"
        mlmodel.short_description = "TSM / ResNet-18 based model for pneumothorax diagnosis in lung ultrasound (POCUS)"
        mlmodel.version = "0.2"
        
        print('Done Converting')
        
        # Convert the PyTorch model to TorchScript
        # BUG WARNING:  Can't convert BatchNorm1D/2D from TorchScript to CoreML:
        #        https://github.com/apple/coremltools/issues/817
        #        The conversion of both BatchNorm1D and BatchNorm2D fails which is a blocker
        #        for using the script mode and converting dynamic models.
        #
        #    This seems to be based on an underlying PyTorch/Script issue:
        #        https://github.com/apple/coremltools/issues/802
        #        https://github.com/pytorch/pytorch/issues/41674
        #        The ONNX exporterâ€”and the CoreML exporter, which is based on the ONNX
        #        exporterâ€”does not handle modification of module attributes within a forward
        #        graph, because ONNX does not support doing that operation (not sure whether
        #        CoreML does, but I'm guessing it doesn't either).
        #        The primary workaround if you needed to export to ONNX/CoreML is probably to
        #    **    rewrite your model so that it does not call setattr in the forward function.  **
        #print('Converting to TourchScript...')
        #scripted_model = torch.jit.script(model)
        #print('Converting to CoreML...')
        #mlmodel = ct.converters.convert( scripted_model, inputs=[ct.TensorType( shape=input_size )] )
        #print('Done Converting')
        
    #    # Save the converted model.
    #    mlmodel.save("TSMNet_no_metadata.mlmodel")
    #    print('Model (without metadata) saved in CoreML Format')
        
        # set class labels
        
        #labels_json = {"labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]}
        
        #class_labels = [ 'no-sliding', 'sliding', ]
        #labels_json = {"labels": class_labels}
        
        #mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
        
        #mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)
        
        ### For a classifier model, the metadata is specified before model conversion
        
        mlmodel.save("TSMNet_with_metadata.mlmodel")
        mlmodel.save("Kinetics.mlmodel")
        print('Model (with metadata) saved in CoreML Format')

        #classifier_config = ct.ClassifierConfig(class_labels)
        
        # Convert the model using the Unified Conversion API
        #model = ct.convert( keras_model, inputs=[image_input], classifier_config=classifier_config,)