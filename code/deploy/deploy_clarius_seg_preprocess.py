"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-23 15:42:00
 * @modify date 2022-01-07 19:43:33
 * @desc [description]
 """

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



## Model


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

     
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        # offset = outputs2.size()[2] - inputs1.size()[2]
        # print('offset:', offset)
        # padding = 2 * [offset // 2, offset // 2]
        # print('padding size:',padding)
        # outputs1 = F.pad(inputs1, [0, 0, 0, 0])

        # print([inputs1.size(), outputs1.size(), outputs2.size()])
        # return self.conv(torch.cat([outputs1, outputs2], 1))
        return self.conv(torch.cat([inputs1, outputs2], 1))


class unet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=5, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        # return final
        return {"pred_seg": final}


class unetSmall(nn.Module):

    def __init__(self, feature_scale=4, n_classes=5, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unetSmall, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        # self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        # self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        # self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.final = nn.Conv2d(filters[2], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        # up2 = self.up_concat2(conv2, up3)
        # up1 = self.up_concat1(conv1, up2)

        # final = self.final(up1)
        final = self.final(up3)

        # return final
        return {"pred_seg": final}



import cv2

def load_clip(file_name, x_range=None, y_range=None):
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


def load_frames(file_name, x_range=None, y_range=None):
    print(file_name)
    frame_names = [f for f in os.listdir(file_name) if '.png' in f]

    frame_names.sort()

    frames = [cv2.imread(os.path.join(file_name, f)) for f in frame_names]


    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)/255.0 for f in frames]


    return frames


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

SEG_LABEL_NAMES = [ 'bck-grd', 'pleura', 'chest-wall', 'rib', ]
colormap = [
                (0, 0, 0), #0 #000000 - Background
                # (242, 5, 246), #1 #F205F6 - pleural line pneumothorax
                (255, 0, 0), #2 #FF0000 - pleural line normal
                # (0, 255, 0), #3 #00FF00 - vessel
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

    model_path = "/home/grg/Research/DARPA-Pneumothorax/unetSm_DARPA-Seg-Dataset_Frame_Seg_4cls_ISBI_pretrain_T1_A_last.ckpt"

    # video_path = "/home/grg/Research/DARPA-Pneumothorax/NoStreating-33-RawBscan/env"
    video_path = "/home/grg/Research/DARPA-Pneumothorax/NoStreating-33-RawBscan/bscan"

    # Initialize model
    # Input dimensions are NxTxCxHxW = 1x15x1x224x224 (with rectification TxHxW = 15x1080x1920)
   
    model = unetSmall(
                feature_scale = 1, 
                in_channels = 1,
                # n_classes = cfg.DATA.NUM_CLASS,
                n_classes = 4,
            )


    #Load model weights
    # CHANGE
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    #Fix keys
    state_dict = {}
    for k,v in checkpoint["state_dict"].items():
        state_dict[".".join(k.split(".")[1:])] = v
    
    model.load_state_dict(state_dict)
    
    # Put network in inference (evaluation) mode
    model.eval()
    
    
    #Load Video
    print('loading video')

    video_frames = load_frames(video_path)
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
    #Resize frames
    print('resizing video')
    image_size = (224, 224)

    video_frames = [cv2.resize(f, image_size) for f in video_frames]


    #Model Forward pass
    print('Evaluating model')
    class_labels = [ 'no-sliding', 'sliding', ]
    
    with torch.no_grad():
        
        for idx in range(len(video_frames)):
            print('    preparing frame ', idx)
            clip = video_frames[idx]
            
            # Remap clip to have dimensions NxCxHxW:
            clip = torch.tensor(clip).unsqueeze(0).unsqueeze(0).float()
            
            print('    evaluating frame ', idx)
            out_dict = model(clip)

        
            ## Save Seg Pred as GIF
            in_clip = clip.unsqueeze(0)

            in_clip = (in_clip.detach()*255).to(torch.uint8)
            in_clip = in_clip.permute(0,1,3,4,2)
            if in_clip.shape[-1] == 1:
                in_clip = in_clip.repeat(1,1,1,1,3)

            seg_pred = out_dict["pred_seg"]
            # seg_pred = seg_pred.unsqueeze(1)

            seg_pred = seg_pred.argmax(dim = 1, keepdim = True)

            seg_pred = F.interpolate(seg_pred.float(), size = image_size, mode = "nearest").long()

            seg_pred = colormap[seg_pred.detach().cpu().numpy()]


            pred_lb = "NA"

            gif_name = f"input{idx}_unet_seg_pred.gif"
            
            path = "."
            gif_path = os.path.join(path, gif_name)


            gif_video = saveSegPredAsGIF(in_clip.squeeze(0), seg_pred.squeeze(0), pred_lb, gif_path, filename = gif_name)
        
