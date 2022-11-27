import numpy as np
import json

import os

import torch 


def getFeatures(dataset_split_dict, train_folds, test_folds, train_label_dict, test_label_dict, \
    gt_label_dict, label_features, task, SF_category_dict, Disease_category_dict, BiomarkerModel, UsePerPatientFeatures, video_to_patient_wise_info_mapping_dict):



    train_video_list = []
    # [train_video_list.extend(s) for s in dataset_split_dict['train'].values()]
    # [[train_video_list.extend(v) for k, v in fold_data.items() if k in train_folds] for fold_data in dataset_split_dict["train"].values()]
    [train_video_list.extend(dataset_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in train_folds]
    
        
    test_video_list = []
    # [test_video_list.extend(s) for s in dataset_split_dict['test'].values()]
    [test_video_list.extend(dataset_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in test_folds]


    #Map individual video to patient-scan_dates 
    if UsePerPatientFeatures:

        train_video_list = [video_to_patient_wise_info_mapping_dict[v] for v in train_video_list]
        train_video_list = np.unique(train_video_list).tolist()

        test_video_list = [video_to_patient_wise_info_mapping_dict[v] for v in test_video_list]
        test_video_list = np.unique(test_video_list).tolist()

    # #Correct video names
    # train_video_list = [v.split('/')[-1] for v in train_video_list]
    # test_video_list = [v.split('/')[-1] for v in test_video_list]

    # #Exclude some train videos as they are not present in the model - #TODO-GRG: Need to fix this
    # print([ v for v in train_video_list if v not in list(train_label_dict.keys())])
    # # train_video_list = [v for v in train_video_list if v not in ['114/114(06.21.21)L1-120-6(C19, AKI, NSTEMI).avi', '50/50(03.15.21)L1-471-4(NSTEMI, CKD3, obesity).avi', '112/112(06.21.21)R1-300-5(HFpEF, ESRD, GIB).avi', '148/148(08.16.21)L1-192-7(C19, cholecystitis).avi', '21/21(04.25.21)L1-139-7(C19, chronic lung disease).avi']]
    # train_video_list = [v for v in train_video_list if v not in ['96/96(05.13.21)L1-388-5(CHF, HTN emergency).avi', '36/36(03.08.21)L1-329-5(biventricular failure, ESRD, HTN).avi', '21/21(03.12.21)R1-163-6(C19, chronic lung disease).avi', '50/50(03.15.21)L1-471-4(NSTEMI, CKD3, obesity).avi', '112/112(06.21.21)L1-300-5(HFpEF, ESRD, GIB).avi', '148/148(08.16.21)L1-192-7(C19, cholecystitis).avi', '21/21(04.25.21)L1-139-7(C19, chronic lung disease).avi']]
    # train_video_list = [v for v in train_video_list if v not in ['152/152(8.20.21)L1.avi', '61/61(03.22.21)R1-467-4(NSTEMI, COPD, HFpEF).avi', '114/114(06.21.21)L1-120-6(C19, AKI, NSTEMI).avi', '89/89(04.28.21)L1-327-5(CHF, CKD4, AS).avi', '123/123(07.09.21)R1-323-5(hypercapnic resp fail).avi', '21/21(03.22.21)R1-165-7(C19, chronic lung disease).avi']]
    # train_video_list = [v for v in train_video_list if v not in ['152(8.20.21)L1.avi', '61(03.22.21)R1-467-4(NSTEMI, COPD, HFpEF).avi', '114(06.21.21)L1-120-6(C19, AKI, NSTEMI).avi', '89(04.28.21)L1-327-5(CHF, CKD4, AS).avi', '123(07.09.21)R1-323-5(hypercapnic resp fail).avi', '21(03.22.21)R1-165-7(C19, chronic lung disease).avi']]
    
    if BiomarkerModel:
        # train_label = []
        # [[train_label.append(train_label_dict[video][f]) for f in label_features] for video in train_video_list]
        # train_label_ft = np.array([[train_label_dict[video][f] for f in label_features] for video in train_video_list])
        train_label_ft = np.array([np.hstack([train_label_dict[video][f] for f in label_features]) for video in train_video_list])
        assert train_label_ft.shape == (len(train_video_list), 38)
        # train_label = train_label_ft.reshape(train_label_ft.shape[0], -1)
    else:
        train_label_ft = np.array([train_label_dict[video] for video in train_video_list])
        assert train_label_ft.shape == (len(train_video_list), 512)

    if BiomarkerModel:
        # test_label = []
        # [[test_label_dict[video][f] for f in label_features] for video in test_video_list]
        # test_label_ft = np.array([[test_label_dict[video][f] for f in label_features] for video in test_video_list])
        test_label_ft = np.array([np.hstack([test_label_dict[video][f] for f in label_features]) for video in test_video_list])
        assert test_label_ft.shape == (len(test_video_list), 38)
        # test_label = test_label_ft.reshape(test_label_ft.shape[0], -1)
    else:
        test_label_ft = np.array([test_label_dict[video] for video in test_video_list])
        assert test_label_ft.shape == (len(test_video_list), 512)

    gt_train_scores = []
    if task == "sf-ratio":
        gt_train_scores = np.array([SF_category_dict[v] for v in train_video_list])

        assert np.array_equal(np.unique(gt_train_scores), np.arange(4)), "Error! Wrong no of classes."
    elif task == "diagnosis":
        # labels = np.array([self.Disease_category_dict[v] for v in self.videosToIncludeList])

        #Map the 11 disease category to 7 categories
        disease_mapping_dict = {0: 0, 1: 1, 2: 5, 3: 4, 4: 5, 5: 5, 6: 2, 7: 2, 8: 3, 9: 6, 10: 5, 11: 5}
        
        # #Multi-label
        # """
        # #Note: As some videos are marked for multiple disease we are picking the first disease,
        # #  while considering upsampling. We can further optimize this
        # """
        # #TODO-GRG: Need to fix this - i.e. handle muilti-label classification
        # gt_train_scores = np.array([Disease_category_dict[v] if isinstance(Disease_category_dict[v], int) else Disease_category_dict[v][0] for v in train_video_list])

        # gt_train_scores = np.array([disease_mapping_dict[i] for i in gt_train_scores])

        #Multi-class (not multi-label)
        gt_train_scores = np.array([disease_mapping_dict[Disease_category_dict[v]] for v in train_video_list])
   
        assert np.array_equal(np.unique(gt_train_scores), np.arange(7)), "Error! Wrong no of classes."
    else:
    
        [gt_train_scores.append(gt_label_dict[video]['lung-severity']) for video in train_video_list]
        # assert np.array(gt_train_scores).shape == (len(train_video_list), 4)
        gt_train_scores = np.array(gt_train_scores).argmax(1)

        assert np.array_equal(np.unique(gt_train_scores), np.arange(4)), "Error! Wrong no of classes."


    assert gt_train_scores.shape[0] == len(train_video_list)

    gt_test_scores = []
    if task == "sf-ratio":
        gt_test_scores = np.array([SF_category_dict[v] for v in test_video_list])

        assert np.array_equal(np.unique(gt_test_scores), np.arange(4)), "Error! Wrong no of classes."
    elif task == "diagnosis":
        # labels = np.array([self.Disease_category_dict[v] for v in self.videosToIncludeList])
        
        #Map the 11 disease category to 7 categories
        disease_mapping_dict = {0: 0, 1: 1, 2: 5, 3: 4, 4: 5, 5: 5, 6: 2, 7: 2, 8: 3, 9: 6, 10: 5, 11: 5}

        # #Multi-label
        # """
        # #Note: As some videos are marked for multiple disease we are picking the first disease,
        # #  while considering upsampling. We can further optimize this
        # """
        # #TODO-GRG: Need to fix this - i.e. handle muilti-label classification
        # gt_test_scores = np.array([Disease_category_dict[v] if isinstance(Disease_category_dict[v], int) else Disease_category_dict[v][0] for v in test_video_list])
        
        # gt_test_scores = np.array([disease_mapping_dict[i] for i in gt_test_scores])

        #Multi-class (not multi-label)
        gt_test_scores = np.array([disease_mapping_dict[Disease_category_dict[v]] for v in test_video_list])
   
        assert np.array_equal(np.unique(gt_test_scores), np.arange(7)), "Error! Wrong no of classes."
    else:
        [gt_test_scores.append(gt_label_dict[video]['lung-severity']) for video in test_video_list]
        # assert np.array(gt_test_scores).shape == (len(test_video_list), 4)
        gt_test_scores = np.array(gt_test_scores).argmax(1)

        assert np.array_equal(np.unique(gt_test_scores), np.arange(4)), "Error! Wrong no of classes."

    assert gt_test_scores.shape[0] == len(test_video_list)

    # gt_train_label_ft = np.array([[gt_label_dict[video][f] for f in label_features] for video in train_video_list])
    gt_train_label_ft = np.array([np.hstack([gt_label_dict[video][f] for f in label_features]) for video in train_video_list])
    assert gt_train_label_ft.shape == (len(train_video_list), 38)
    # gt_train_label = gt_train_label_ft.reshape(gt_train_label_ft.shape[0], -1)

    # gt_test_label_ft = np.array([[gt_label_dict[video][f] for f in label_features] for video in test_video_list])
    gt_test_label_ft = np.array([np.hstack([gt_label_dict[video][f] for f in label_features]) for video in test_video_list])
    assert gt_test_label_ft.shape == (len(test_video_list), 38)
    # gt_test_label = gt_test_label_ft.reshape(gt_test_label_ft.shape[0], -1)

    #Cal feature accuracy for Biomarker models
    if BiomarkerModel:
        # train_label_ft_acc = (train_label_ft == gt_train_label_ft).mean(axis = (0, 2))
        train_label_ft_acc = (train_label_ft == gt_train_label_ft).mean(axis = 0)
        # print(f"Train feature acc = {[(f,v) for f, v in zip(label_features, train_label_ft_acc)]}")
        print(f"Train feature acc = {train_label_ft_acc}")

        # test_label_ft_acc = (test_label_ft == gt_test_label_ft).mean(axis = (0, 2))
        test_label_ft_acc = (test_label_ft == gt_test_label_ft).mean(axis = 0)
        # print(f"Test feature acc = {[(f,v) for f, v in zip(label_features, test_label_ft_acc)]}")
        print(f"Test feature acc = {test_label_ft_acc}")

    return train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list


def getModelFeaturePreds(dataset_split_dict, exp_path, exp_name, label_dict_path, train_folds, test_folds, TRAIN = False, multi_task = False, use_probs = False):
        
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

    assert prob_preds.shape[1] == 38, "Error! More than 25 biomarker output detected"

    preds = np.load(preds_path)
    filenames = np.load(filenames_path)

    video_list = []
    if TRAIN:
        # [[video_list.extend(v) for k, v in fold_data.items() if k in train_folds] for fold_data in dataset_split_dict[dataset_type].values()]
        [video_list.extend(dataset_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in train_folds]
    else:
        [video_list.extend(dataset_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in test_folds]

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
            # pred = torch.sigmoid(torch.tensor(prob_preds[pred_idx]).mean(0)).numpy()
            pred = torch.sigmoid(torch.tensor(prob_preds[pred_idx]).max(0)[0]).numpy()
        else:
            # pred = (torch.sigmoid(torch.tensor(prob_preds[pred_idx])).mean(0) > 0.5).numpy().astype(np.int)
            pred = (torch.sigmoid(torch.tensor(prob_preds[pred_idx])).max(0)[0] > 0.5).numpy().astype(np.int)
            # pred = preds[pred_idx].mean(0).astype(np.int)

        video_label = {}

        video_label['alines'] = pred[0:5].tolist()

        video_label['blines'] = pred[5:10].tolist()

        video_label['blines_origin'] = pred[10:13].tolist()

        video_label['pleural_thickness'] = pred[13:17].tolist()

        video_label['pleural_location'] = pred[17:20].tolist()

        video_label['pleural_indent'] = pred[20:25].tolist()

        video_label['pleural_break'] = pred[25:30].tolist()

        video_label['consolidation'] = pred[30:35].tolist()

        # video_label['effusion'] = [0, 0, 0, 0, 0]
        video_label['effusion'] = pred[35:38].tolist()

        video_label['lung-severity'] = [1, 0, 0, 0]

        video_label['unusual_findings'] = ''

        label_dict[video] = video_label
        # label_dict[video.split('/')[-1]] = video_label


    #Save splits
    label_json = json.dumps(label_dict, indent=4)
    f = open(label_dict_path,"w")
    f.write(label_json)
    f.close()

    print(f"label_json: {label_json}")


def generateModelFeaturePreds(dataset_split_dict_path, exp_path, exp_name, label_dict_path, train_folds, test_folds, multi_task = False, use_probs = False):


    with open(dataset_split_dict_path, 'r') as json_file:
        dataset_split_dict = json.load(json_file)

    getModelFeaturePreds(dataset_split_dict, exp_path, exp_name, label_dict_path, train_folds, test_folds, TRAIN = True, multi_task = multi_task, use_probs = use_probs)
    getModelFeaturePreds(dataset_split_dict, exp_path, exp_name, label_dict_path, train_folds, test_folds, TRAIN = False, multi_task = multi_task, use_probs = use_probs)


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
    
    
    