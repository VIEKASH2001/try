"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-15 21:46:01
 * @modify date 2021-11-18 17:38:25
 * @desc [description]
 """
 
import numpy as np
import pandas as pd

import os

import json


if __name__ == "__main__":

    # config_path = "/data1/datasets/DARPA POCUS AI/Original Datasets/POCUS AI Phase 1 Training Data (9-15-21) - with_difficulty_labels_n_data_splits.xlsx"
    config_path = "/data1/datasets/DARPA-Dataset/POCUS AI Phase 1 Training Data (9-15-21) - with_difficulty_labels_n_data_splits.xlsx"
    
    datadir = "/data1/datasets/DARPA-Dataset/"

    # config_pd = pd.read_excel(config_path)
    config_pd = pd.ExcelFile(config_path)
    config_nosliding = pd.read_excel(config_pd, 'No-Sliding')
    config_sliding = pd.read_excel(config_pd, 'Sliding')

    assert np.array_equal(config_sliding.columns, config_nosliding.columns), "Error! Column names mismatch between sliding & no-sliding sheets."
    header = list(config_sliding.columns)

    config_nosliding = np.array(config_nosliding)
    config_sliding = np.array(config_sliding)

    #Remove NaN rows
    config_nosliding = np.array([a for a in config_nosliding if  type(a[0]) == int or type(a[0]) == str or (type(a[0]) == float and not np.isnan(float(a[0])))])
    config_sliding = np.array([a for a in config_sliding if  type(a[0]) == int or type(a[0]) == str or (type(a[0]) == float and not np.isnan(float(a[0])))])

    assert config_nosliding.shape[0] == 30 and config_sliding.shape[0] == 32, "Error! Wrong number of samples"

    videos = []
    videos += ["NoSliding/"+ ".".join(v.split('.')[:-1]).replace('image_', '') for v in config_nosliding[:, header.index("File name")]]
    videos += ["Sliding/"+ ".".join(v.split('.')[:-1]).replace('image_', '') for v in config_sliding[:, header.index("File name")]]

    #Sort videos
    videos.sort()
    
    ## Create processed dataset files
    
    # video_type = "crop_image"
    video_type = "crop_image_rct"
    # video_type = "linear_probe_under_pleural_line"

    processed_dataset_json_path = os.path.join(datadir, f"processed_dataset_files_{video_type}.json")

    processed_dataset_dict = {}
    if os.path.exists(processed_dataset_json_path):
        
        with open(processed_dataset_json_path, 'r') as json_file:
            processed_dataset_dict = json.load(json_file)

    for video in videos:

        video_dir = os.path.join(datadir, video, video_type)

        frame_filenames = [f for f in os.listdir(video_dir) if '.png' in f or '.jpeg' in f or '.jpg' in f or '.JPG' in f]
        frame_filenames.sort()

        if video in processed_dataset_dict:

            processed_frames = list(processed_dataset_dict[video].keys())
            processed_frames.sort()

            if np.array_equal(frame_filenames, processed_frames):
                continue
            else:
                raise Exception(f"Frames mismatch for video = {video}!")

        else:
            print(f"Processing video : {video}")

            processed_frame_dict = {}
            for f in frame_filenames:
                processed_frame_dict[f] = os.path.join(video_dir, f) 

            processed_dataset_dict[video] = processed_frame_dict



        #Save processed dataset dict
        processed_dataset_json = json.dumps(processed_dataset_dict, indent=4)
        f = open(processed_dataset_json_path, "w")
        f.write(processed_dataset_json)
        f.close()

    
    ## Create dataset split json

    random_split_trial = "R1"

    split_json_path = os.path.join(datadir, f'dataset_split_equi_class_{random_split_trial}.json')

    folds_list = ["A", "B", "C"]
    folds = len(folds_list)

    lung_severity = ["NoSliding", "Sliding"]


    #Write split info to file
    infoFile = open(split_json_path.split(".")[0] + "_info.txt","w")
    infoFile.write(f"Generating data split : {split_json_path}\n\n")


    train_video_dict = {}
    test_video_dict = {}


    #Process all severity images
    for severity, config in [("NoSliding", config_nosliding), ("Sliding", config_sliding)]:
        
        video_list = [severity + "/" + v.replace('image_', '') for v in config[:, header.index("File name")]]

        test_videos = [severity + "/" + v.replace('image_', '') for i, v in enumerate(config[:, header.index("File name")]) if config[i, header.index("Data Split")] == "T"]
             
        n = len(video_list)

        no_test_videos = len(test_videos)

        no_train_videos = n - no_test_videos
 
        print(f"{severity} : Total no of Videos = {n}; [Train = {no_train_videos}; Test = {no_test_videos}]")
        infoFile.write(f"{severity} : Total no of Videos = {n}; [Train = {no_train_videos}; Test = {no_test_videos}]\n")

        test_video_dict[str(severity)] = test_videos

        train_folds_dict = {}
        for fold_id in range(folds):
            train_folds_dict[folds_list[fold_id]] = [severity + "/" + v.replace('image_', '') for i, v in enumerate(config[:, header.index("File name")]) if config[i, header.index("Data Split")] == folds_list[fold_id]]
             
        assert no_train_videos == sum([len(v) for v in train_folds_dict.values()]), "Error! Train fold video count mismatch."
       
        train_video_dict[str(severity)] = train_folds_dict

        assert no_test_videos == len(test_video_dict[str(severity)])
        assert no_train_videos == sum([len(l) for l in train_video_dict[str(severity)].values()])

        print(f"{severity} : Train-folds - {[len(l) for l in train_video_dict[str(severity)].values()]}")
        infoFile.write(f"{severity} : Train-folds - {[len(l) for l in train_video_dict[str(severity)].values()]}\n\n")

    #Close split info file
    infoFile.close()

    test_pts = np.unique(np.hstack([[v.split('(')[0] for v in v_list] for v_list in test_video_dict.values()]))

    train_pts = []
    [[train_pts.extend([v.split('(')[0] for v in v_list]) for v_list in l.values()] for l in train_video_dict.values()]
    train_pts = np.unique(train_pts)

    for t_pt in test_pts:
        assert t_pt not in train_pts, f"Error! test patient {t_pt} is in train set"

    split_dict = {"train": train_video_dict, "test": test_video_dict}

    #Save splits
    split_json = json.dumps(split_dict, indent=4)
    f = open(split_json_path,"w")
    f.write(split_json)
    f.close()


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


