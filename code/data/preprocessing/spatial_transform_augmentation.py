import numpy as np
import torch

import cv2 

import sys
sys.path.append("./code")

from analysis.visualizeAnnotation import saveInputAsGIF

import torch.nn.functional as F

import torchvision.transforms.functional_tensor as F_t



def spatialTransformAugClip(clip):
    
    t, c, h, w = clip.shape

    #Scale params
    sx, sy = np.random.uniform(low=0.8, high=1.3, size=(2,))
    # sx, sy = (1,1)

    #Scale params
    # tx, ty = np.random.uniform(low=-0.05, high=0.05, size=(2,))
    tx, = np.random.uniform(low=-0.05, high=0.05, size=(1,))
    ty, = np.random.uniform(low=-0.02, high=0.08, size=(1,))    
    # tx, ty = (0,0) #(-0.2,0.2)

    #Rotation angle
    alpha = (np.pi/360.0)*np.random.uniform(low=-15, high=15, size=(1,))
    # alpha = (np.pi/360.0)*5

    R_scale = np.array([[sx,0],[0,sy]])
    R_rot = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]]).squeeze()
    T = np.array([tx,ty])

    theta = np.hstack((np.matmul(R_rot, R_scale),T[:,np.newaxis]))

    # Repeat the SAME transform tensor for all time-frames
    theta = torch.tensor(theta).float().unsqueeze(0).repeat(t,1,1)
    # # theta = theta.view(-1, 3, 4)
    # theta = theta.view(-1, 2, 3)

    # print(f"theta: {theta.detach().cpu().numpy()}")

    grid = F.affine_grid(theta, clip.size(), align_corners = True)
    aug_clip = F.grid_sample(clip, grid, align_corners = True)

    #Crop and resize
    cx, cy = (25, 25)
    aug_clip = aug_clip[:,:,cy:-cy,cx:-cx]

    # # Resize the images
    aug_clip = F.interpolate(aug_clip, size = (h,w), mode = "bicubic", align_corners = True)
    # clip = (clip - clip.min())/(clip.max() - clip.min())
    aug_clip = aug_clip.clamp(min=0.0, max=1.0) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)


    return aug_clip


def gaussianNoise(clip):

    # sigma, = np.random.uniform(low=0.1, high=2, size=(1,)).tolist()
    # sigma, = np.random.uniform(low=0.001, high=0.005, size=(1,)).tolist()
    sigma, = np.random.uniform(low=0.0001, high=0.0005, size=(1,)).tolist()
    clip = clip + (sigma**0.5)*torch.randn(clip.shape)
    clip = clip.clamp(min=0.0, max=1.0)
    return clip


def gaussianBlur(clip):

    # kernel_sz_x, kernel_sz_y = np.random.choice([3,5,7], 2).tolist()
    # sigma_x, sigma_y = np.random.uniform(low=0.1, high=2, size=(2,)).tolist()
    kernel_sz_x, kernel_sz_y = np.random.choice([3,5], 2).tolist()
    sigma_x, sigma_y = np.random.uniform(low=0.1, high=1, size=(2,)).tolist()    
    clip = F_t.gaussian_blur(clip, [kernel_sz_x, kernel_sz_y], [sigma_x, sigma_y])

    return clip

if __name__ == "__main__":


    path = "/data1/datasets/LSU-Dataset/21/21(03.05.21)L1-184-7(C19 long hauler 3wks)/crop_image"

    clip_path = "input_clip.gif"
    aug_clip_path = "aug_clip.gif"

    clip_frame_ids = np.arange(1, 160, 10)

    clip = torch.Tensor([cv2.resize(cv2.imread(f"{path}/frame{f}.png"), (224,224)) for f in clip_frame_ids]) #Dim - TxHxWxC
    clip = clip.permute(0,3,1,2) #NxHxWxC -> NxCxHxW
    clip = clip[:,0].unsqueeze(1) #Convert RGB to Grey as this dataset is Grey scale #TODO-GRG : Note this needs to be changed for other datasets

    # # Resize the images
    # clip = F.interpolate(clip, size = self.image_size, mode = "bicubic", align_corners = True)
    # # clip = (clip - clip.min())/(clip.max() - clip.min())
    # clip = clip.clamp(min=0, max=255) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)

    #Convert image from 0-255 to 0-1 range
    clip = clip/255.0


    #Save input clip
    saveInputAsGIF(clip.permute(0, 2, 3, 1).numpy(), clip_path)
    
    for i in range(10):

        aug_clip = spatialTransformAugClip(clip)

        #Save augmented clip
        saveInputAsGIF(aug_clip.permute(0, 2, 3, 1).numpy(), f"aug_clip{i}.gif")

        aug_clip = gaussianBlur(aug_clip)

        saveInputAsGIF(aug_clip.permute(0, 2, 3, 1).numpy(), f"aug_gb_clip{i}.gif")

        aug_clip = gaussianNoise(aug_clip)

        saveInputAsGIF(aug_clip.permute(0, 2, 3, 1).numpy(), f"aug_gn_clip{i}.gif")