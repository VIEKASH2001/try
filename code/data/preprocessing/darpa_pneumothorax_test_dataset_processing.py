"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-15 21:46:01
 * @modify date 2021-11-18 17:50:52
 * @desc [description]
 """
 
import numpy as np
import pandas as pd

import os

import json


if __name__ == "__main__":

    # config_path = "/data1/datasets/DARPA POCUS AI/Original Datasets/POCUS AI Phase 1 Training Data (9-15-21) - with_difficulty_labels_n_data_splits.xlsx"
    # config_path = "/data1/datasets/DARPA-Dataset/POCUS AI Phase 1 Training Data (9-15-21) - with_difficulty_labels_n_data_splits.xlsx"
    
    datadir = "/data1/datasets/DARPA-Dataset/"

  
    videos = []
    videos += ["NoSliding/" + v for v in os.listdir(os.path.join(datadir, "NoSliding")) if os.path.isdir(os.path.join(datadir, "NoSliding", v))]
    videos += ["Sliding/" + v for v in os.listdir(os.path.join(datadir, "Sliding")) if os.path.isdir(os.path.join(datadir, "Sliding", v))]

    #Sort videos
    videos.sort()
    
    ## Create processed dataset files
    
    # video_type = "crop_image"
    video_type = "crop_image_rct"
    # video_type = "linear_probe_under_pleural_line"

    
    ## Create dataset labels json

    label_path = os.path.join(datadir, f"darpa_labels.json")
    label_feature = "sliding"

    video_labels_dict = {}

    for video in videos:
        
        feature_dict = {}
        
        if "NoSliding" not in video:
            feature_dict[label_feature] = [0, 1]
        else:
            feature_dict[label_feature] = [1, 0]
        
        video_labels_dict[video + ".mp4"] = feature_dict


    #Save splits
    video_labels_json = json.dumps(video_labels_dict, indent=4)
    f = open(label_path,"w")
    f.write(video_labels_json)
    f.close()


