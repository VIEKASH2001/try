"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-23 15:42:00
 * @modify date 2021-11-23 16:58:25
 * @desc [description]
 """

import torch
import torch.nn.functional as F

import tqdm

import os

import cv2

from model.unet import unet

num_class = 8
labels = ['padding_mIoU', 'background_mIoU', 'a_lines_mIoU', 'b_lines_mIoU', 'healthy_plural_line_mIoU', 'unhealthy_plural_line_mIoU', 'healthy_region_mIoU', 'unhealthy_region_mIoU']

device = "cuda:0"

showPlot = False

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
Lung_Colors =  np.array([
    [0, 0, 0, 255], #Padded region
    [237, 164, 118, 255], #SandyBrown - Background
    [255, 255, 0, 255], #Yellow - A-lines
    # [35, 98, 143, 255], #DarkBlue - B-line
    [115, 115, 115, 255], #DarkBlue - B-line
    # [106, 255, 0, 255], #Green - Healthy Plural Line
    [79, 143, 35, 255], #DarkGreen - Healthy Plural Line
    # [255, 0, 0, 255], #Red - Unhealthy Plural Line
    [255, 127, 0, 255], #Orange - Unhealthy Plural Line
    [185, 237, 224, 255], #LightBlue - Healthy Region
    [237, 185, 185, 255] #Pink - Unhealthy Region
    ])/255  
cmap_name = ListedColormap(Lung_Colors.tolist())

if __name__ == "__main__":

    # dataset_path = "/data1/datasets/DARPA-Dataset/NoSliding"
    dataset_path = "/data1/datasets/DARPA-Dataset/Sliding"

    model_path = "/home/grg/Research/LungUS-AI/results/ISBI/Exp1_UNet_Dense_A/best_model.pth"

    model = unet(feature_scale = 1, in_channels = 1, n_classes = num_class)

    #Load model weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    model = model.to(device)
    model.eval()

    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    folders.sort()

    num_folders = len(folders)

    with torch.no_grad():
        
        for i, folder in tqdm.tqdm(enumerate(folders)):
            
            print(f"[{i}/{num_folders}] {folder}")

            image_folder = os.path.join(dataset_path, folder, 'crop_image_rct')
            image_names = os.listdir(image_folder)

            image_names.sort(key=lambda x:int(x.split(".")[0]))

            save_path = os.path.join(dataset_path, folder, "segmentations")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for j, image_file in tqdm.tqdm(enumerate(image_names)): 
                

                image_org = cv2.imread(os.path.join(image_folder, image_file))

                # image = cv2.resize(image, (2*(image.shape[1]//2), 2*(image.shape[0]//2)))
                image = cv2.resize(image_org, (624, 464))
                image = image[:,:,0] / 255.0

                image = torch.tensor(image, device = device, dtype = torch.float32).unsqueeze(0).unsqueeze(1)

                output = model(image).squeeze(0)

                output_prob = F.softmax(output, dim = 0)
            
                output_cls = output_prob.argmax(0)

                np.savez(os.path.join(save_path, image_file.split(".")[0]), output.detach().cpu().numpy())

                output_img = cv2.resize(output_cls.detach().cpu().numpy(), image_org.T.shape[1:], interpolation = cv2.INTER_NEAREST)
                output_img = Lung_Colors[output_img]
                output_img = 255*output_img[:,:,::-1][:,:,1:]
                cv2.imwrite(os.path.join(save_path, image_file), output_img)
                # plt.imshow(output_cls.detach().cpu().numpy(), vmin = 0, vmax = num_class, cmap=plt.get_cmap(cmap_name))
                # plt.savefig(os.path.join(save_path, image_file))

                if showPlot:

                    fig, axs = plt.subplots(1,2)
                    axs[0].imshow(image.squeeze().detach().cpu().numpy())

                    axs[1].imshow(output_cls.detach().cpu().numpy(), vmin = 0, vmax = num_class, cmap=plt.get_cmap(cmap_name))
                    plt.show()

        
