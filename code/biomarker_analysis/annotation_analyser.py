
import numpy as np
from PIL import Image, ImageTk
import cv2

import os
import sys

import time

import json




def main():

    # rootDir = "./"
    # rootDir = "/datahd/GRG/OCT_Ultrasound_Segmentation/Dataset_LSU/Big data dump/"
    # rootDir = "/datahd/GRG/OCT_Ultrasound_Segmentation/Dataset_LSU/Big data dump/score-5"
    rootDir = "/datahd/GRG/OCT_Ultrasound_Segmentation/Dataset_LSU/Big data dump/videos"

    label_filename = "./user_label_ggare_0.json"

    with open(label_filename, 'r') as outfile:
        label_dict = json.load(outfile)

    stat_dict = {}


    nih_stat_dict = {}

    for video in label_dict.keys():

        sf_ratio = int(video.split('-')[1])

        nih_score = int(video.split('-')[2].split('(')[0])


        video_label = label_dict[video]

        lung_sev = np.argmax(video_label['lung-severity'])

        if lung_sev in stat_dict:
            stat_dict[lung_sev] = stat_dict[lung_sev] + [sf_ratio]
        else:
            stat_dict[lung_sev] = [sf_ratio]


        if lung_sev in nih_stat_dict:
            nih_stat_dict[lung_sev] = nih_stat_dict[lung_sev] + [nih_score]
        else:
            nih_stat_dict[lung_sev] = [nih_score]


    # for lung_sev in stat_dict.keys():
    for lung_sev in range(4):
        print(f"Lung severity score-{lung_sev} : mean SF ratio {np.array(stat_dict[lung_sev]).mean()} +/- {np.array(stat_dict[lung_sev]).std()}")


    for lung_sev in nih_stat_dict.keys():
        print(f"Lung severity score-{lung_sev} : mean NIH score {np.array(nih_stat_dict[lung_sev]).mean()} +/- {np.array(nih_stat_dict[lung_sev]).std()}")

    pass

if __name__ == "__main__":
    print("Started...")    
    main()
    print("Finsihed!")