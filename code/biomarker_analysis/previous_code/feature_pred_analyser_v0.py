import numpy as np
import json

import os

import torch 


def getModelFeaturePreds(dataset_split_dict, exp_path, exp_name, label_dict_path, train_folds, TRAIN = False, multi_task = False, use_probs = False):
        
    targets_path = os.path.join(exp_path, exp_name, "Test_targets.npy")
    prob_preds_path = os.path.join(exp_path, exp_name, "Test_prob_preds.npy")
    preds_path = os.path.join(exp_path, exp_name, "Test_preds.npy")
    filenames_path = os.path.join(exp_path, exp_name, "Test_filenames.npy")
    
    
    if TRAIN:
        label_dict_path = label_dict_path.replace("_test.json", "_train.json")
        
    if TRAIN:
        targets_path = targets_path.replace('Test_', 'Train_')
        prob_preds_path = prob_preds_path.replace('Test_', 'Train_')
        preds_path = preds_path.replace('Test_', 'Train_')
        filenames_path = filenames_path.replace('Test_', 'Train_')


    dataset_type = 'test'
    if TRAIN:
        dataset_type = 'train'

    targets = np.load(targets_path)
    
    prob_preds = np.load(prob_preds_path)
    # If multi-task remove the lung-severity neurons
    if multi_task:
        prob_preds = prob_preds[:,4:]

    assert prob_preds.shape[1] == 25, "Error! More than 25 biomarker output detected"

    preds = np.load(preds_path)
    filenames = np.load(filenames_path)

    video_list = []
    if TRAIN:
        [[video_list.extend(v) for k, v in fold_data.items() if k in train_folds] for fold_data in dataset_split_dict[dataset_type].values()]
    else:
        [video_list.extend(s) for s in dataset_split_dict[dataset_type].values()]

    if TRAIN:
        # video_list = [v for v in video_list if ".".join(v.split("/")[-1].split(".")[:-1]) not in np.unique(filenames).tolist()] #Exluded videos
        print([v for v in video_list if ".".join(v.split("/")[-1].split(".")[:-1]) not in np.unique(filenames).tolist()]) #Exluded videos
        video_list = [v for v in video_list if ".".join(v.split("/")[-1].split(".")[:-1]) in np.unique(filenames).tolist()] #Included videos
    assert len(video_list) == np.unique(filenames).shape[0], "No of videos mismatch!"
    
    label_dict = {}
    for idx, video in enumerate(video_list):
        
        #Take multi-clip consensus
        pred_idx = np.where(filenames == ".".join(video.split("/")[-1].split(".")[:-1]))


        if use_probs:
            pred = torch.sigmoid(torch.tensor(prob_preds[pred_idx]).mean(0)).numpy()
        else:
            pred = (torch.sigmoid(torch.tensor(prob_preds[pred_idx])).mean(0) > 0.5).numpy().astype(np.int)
            # pred = preds[pred_idx].mean(0).astype(np.int)

        video_label = {}

        video_label['alines'] = pred[0:5].tolist()

        video_label['blines'] = pred[5:10].tolist()

        video_label['pleural_indent'] = pred[10:15].tolist()

        video_label['pleural_break'] = pred[15:20].tolist()

        video_label['consolidation'] = pred[20:25].tolist()

        # video_label['effusion'] = [0, 0, 0, 0, 0]
        video_label['effusion'] = [0, 0, 0]

        video_label['lung-severity'] = [1, 0, 0, 0]

        video_label['unusual_findings'] = ''

        # label_dict[video] = video_label
        label_dict[video.split('/')[-1]] = video_label


    #Save splits
    label_json = json.dumps(label_dict, indent=4)
    f = open(label_dict_path,"w")
    f.write(label_json)
    f.close()

    print(f"label_json: {label_json}")


def generateModelFeaturePreds(dataset_split_dict_path, exp_path, exp_name, label_dict_path, train_folds, multi_task = False, use_probs = False):


    with open(dataset_split_dict_path, 'r') as json_file:
        dataset_split_dict = json.load(json_file)

    getModelFeaturePreds(dataset_split_dict, exp_path, exp_name, label_dict_path, train_folds, TRAIN = True, multi_task = multi_task, use_probs = use_probs)
    getModelFeaturePreds(dataset_split_dict, exp_path, exp_name, label_dict_path, train_folds, TRAIN = False, multi_task = multi_task, use_probs = use_probs)


if __name__ == '__main__':


    dataset_split_dict_path = "/data1/datasets/LSU-Dataset/dataset_split_equi_class_R1.json"
    
    exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps"
    
    # exp_name = "tsm_LSU-Dataset_AP_RandTS_large_T1_A"
    # exp_name = "tsm_LSU-Dataset_AP_RandTS_PlMl_T1_B"
    # exp_name = "tsm_LSU-Dataset_AP_RandTS_PlMl_T1_A"
    # exp_name = "tsm_LSU-Dataset_AP_EquiTS_PlMl_T1_A"
    # exp_name = "tsm_LSU-Dataset_Savg_EquiTS_PlMl_T1_B"
    # exp_name = "tsm_LSU-Dataset_Savg_EquiTS_PlMl_T1_A"
    # exp_name = "tsm_LSU-Dataset_AP_RN34_EquiTS_PlMl_T1_A"
    # exp_name = "tsm_LSU-Dataset_Test_T1_B"
    # exp_name = "tsm_LSU-Dataset_MultiTask_AP_EquiTS_PlMl_Epoch50_T1_A"
    # exp_name = "tsm_LSU-Dataset_MultiTask_AP_EquiTS_PlMl_Epoch50_T2_A"
    exp_name = "tsm_LSU-Dataset_MultiTask_AP_EquiTS_PlMl_T2_A"
    
    train_folds = ["B", "C"]
    # train_folds = ["A", "C"]
    
    label_dict_path = 'user_label_tsmNet_8a.json'
    # label_dict_path = 'user_label_tsmNet_7a.json'
    # label_dict_path = 'user_label_tsmNet_6a.json'
    # label_dict_path = 'user_label_tsmNet_4a.json'
    # label_dict_path = 'user_label_tsmNet_5a.json'
    # label_dict_path = 'user_label_tsmNet_4b.json'
    # label_dict_path = 'user_label_tsmNet_3a.json'
    # label_dict_path = 'user_label_tsmNet_2a.json'
    # label_dict_path = 'user_label_tsmNet_1.json'

    generateModelFeaturePreds(dataset_split_dict_path, exp_path, exp_name, label_dict_path, train_folds)
    
    
    