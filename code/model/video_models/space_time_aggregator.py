"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-12 01:38:24
 * @modify date 2022-02-24 19:09:18
 * @desc [description]
 """
 

import torch
from torch import nn
import torch.nn.functional as F


class SavgDropClsTavg(torch.nn.Module):

    def __init__(self, fc, dprob, num_segments, num_features, dim=1):
        super(SavgDropClsTavg, self).__init__()
        
        self.dim = dim
        self.num_segments = num_segments
        self.num_features = num_features
        self.dprob = dprob

        self.module = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1), 
                        nn.Dropout(p=self.dprob),
                        nn.Flatten(),
                        fc
                    )

    def forward(self, input_tensor):

        nt, chw = input_tensor.size()

        h = int((chw/self.num_features)**0.5)

        # NTxCHW -> NTxCxHxW
        input_tensor = input_tensor.reshape(nt, self.num_features, h, h)

        # NTxCxHxW -> NTxClass
        output = self.module(input_tensor)     

        # NTxClass -> NxTxClass
        output = output.view((-1, self.num_segments) + output.size()[1:]) 
       
        # NxTxClass -> NxClass
        output = output.mean(dim=self.dim, keepdim=False)

        return output


class SavgDropTmaxCls(torch.nn.Module):

    def __init__(self, fc, dprob, num_segments, num_features, dim=1, return_features = False):
        super(SavgDropTmaxCls, self).__init__()
        
        self.dim = dim
        self.num_segments = num_segments
        self.num_features = num_features
        self.dprob = dprob
        self.return_features = return_features

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
        output_ft = output.max(dim=self.dim, keepdim=False)[0]

        # NxC -> NxClass
        output = self.fc(output_ft)
        
        if self.return_features:
            return output, output_ft
            
        return output


class TmaxSavgDropCls(torch.nn.Module):

    def __init__(self, fc, dprob, num_segments, num_features, dim=1):
        super(TmaxSavgDropCls, self).__init__()
        
        self.dim = dim
        self.num_segments = num_segments
        self.num_features = num_features
        self.dprob = dprob

        self.module = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1), 
                        nn.Dropout(p=self.dprob),
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


'''
Steps for Attention Pooling:
    1) The encoded features F are reshaped as NxTxCxHxW -> NxTHWxC
    2) Adding 1D positional encoding (This can be further enhanced to 2D/3D positional encoding)
    3) Making use of 3 attention heads to model class-specific query similar to multi-headed attention
    4) For every attention head:
        a) Get attention weights A by passing the features F through: Linear layer -> Softmax -> Dropout
        b) Take dot product between the attention weights A with the original features F
        c) Pass the weighted features through a Linear layer -> ReLU
    5) Concat the encoded attention features and pass through a Linear layer to classify into Pneumothorax or Non-pneumothorax
'''
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

        final_features = torch.column_stack(attended_features)
        
        # NxF -> NxC
        final_outputs = self.final_classifiers(final_features)

        return final_outputs


class AttentionPoolingLnorm(nn.Module):
    def __init__(self, num_segments, num_classes, num_features, num_layers = 3, dim=1, height=7, dropout = 0.2):
        super(AttentionPoolingLnorm, self).__init__()

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

        self.layer_norm = nn.LayerNorm(self.num_layers)

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



        # final_features = torch.column_stack(attended_features)
        final_features = torch.stack(attended_features, dim = -1)
        
        #Layer norm - normalize in order to account for the fact that features (pleura vs chest muscle) have different area i.e. # of pixels is different
        final_features = self.layer_norm(final_features)

        # NxCxC' -> NxCC'
        final_features = final_features.reshape(-1, int(self.hidden_dim/4)*self.num_layers)
        
        # # NxF -> NxC
        final_outputs = self.final_classifiers(final_features)

        return final_outputs





class SegAttentionPooling(nn.Module):
    def __init__(self, num_segments, num_classes, num_features, num_layers=3, dim=1, height=56, dropout = 0.2):
        super(SegAttentionPooling, self).__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.num_clips = num_segments
        self.feature_dim = num_features
        self.num_layers = num_layers
        self.hidden_dim = num_features

        self.positional_encoding = PositionalEncoding(self.feature_dim, max_len=self.num_clips*height*height**2)
        
        # self.attention_pooling_layers = nn.ModuleList([nn.Linear(self.hidden_dim, int(self.hidden_dim/4), bias = False) for i in range(self.num_layers)])
        # self.attention_pooling_layers = nn.ModuleList([nn.Linear(self.hidden_dim, 1, bias = False) for i in range(self.num_layers)])
        # self.dropout = nn.Dropout(dropout)

        self.attention_encoder_layer = nn.Sequential(nn.Linear(int(self.hidden_dim*self.num_clips), self.hidden_dim), nn.ReLU())

        self.layer_norm = nn.LayerNorm(self.num_layers)

        self.final_classifiers = nn.Sequential(nn.Linear(int(self.num_layers*self.hidden_dim), self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.num_classes))
        

    def forward(self, features, seg_features):

        seg_softmax_features = F.softmax(seg_features, dim = 1)

        nt, c, h, w = features.size()

        # NTxCxHxW -> NxTHWxC
        features = features.reshape(-1, self.num_clips*h*w, c)

        features = self.positional_encoding(features)

        # NxTHWxC-> NxTxCxHxW
        features = features.reshape(-1, self.num_clips, c, h, w)

        # NTxC'xHxW -> NxTxC'xHxW
        seg_softmax_features = seg_softmax_features.reshape(-1, self.num_clips, self.num_layers, h, w)

        
        attended_features = []
        for feature_idx in range(self.num_layers):
            # attention_pool_output = self.attention_pooling_layers[i](features)
            
            # attention_weights = F.softmax(attention_pool_output.squeeze(-1), dim=-1)
            
            attention_weights = seg_softmax_features[:,:,feature_idx]
            # attention_weights = self.dropout(attention_weights)
            
            # attended_feature = torch.matmul(attention_weights.unsqueeze(2), features)
            #Do Matmul of F(NxTxCxHxW) * A(NxTxHxW) : F(NTxCxHW) * A(NTxHWx1) = AF(NTxCx1)
            attended_feature = torch.bmm(features.reshape(features.shape[0]*self.num_clips, c, -1), attention_weights.reshape(attention_weights.shape[0]*self.num_clips, -1, 1))

            # NTxCx1 -> NxTC
            attended_feature = attended_feature.reshape(-1, self.num_clips*c)

            attended_feature = self.attention_encoder_layer(attended_feature)
            
            attended_features.append(attended_feature)

        # final_features = torch.column_stack(attended_features)
        final_features = torch.stack(attended_features, dim = -1)
        
        #Layer norm - normalize in order to account for the fact that features (pleura vs chest muscle) have different area i.e. # of pixels is different
        final_features = self.layer_norm(final_features)

        # NxCxC' -> NxCC'
        final_features = final_features.reshape(-1, c*self.num_layers)
        
        # NxF -> NxC
        final_outputs = self.final_classifiers(final_features)

        return final_outputs

