import numpy as np

import os

#For saving & loading sklearn model - Ref: https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
from joblib import dump, load


import sys
sys.path.append("./code")
import utils

from biomarker_analysis.feature_pred_analyser import generateModelFeaturePreds, getFeatures

from biomarker_analysis.decision_tree_postprocess import decisonTreePostAnalysis

import biomarker_analysis.ML_classifiers as ml
# import biomarker_analysis.DSO_classifiers as dso



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, multilabel_confusion_matrix
from scipy.special import softmax
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def calScores(preds, prob_preds, targets, class_names, task, logger, binary_cross_entropy = False):

    labels = np.arange(len(class_names))
    

    accuracy = accuracy_score(targets, preds)

    if binary_cross_entropy:
        confusionMatrix = multilabel_confusion_matrix(targets, preds, labels = labels)
    else:
        confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    # confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    if binary_cross_entropy:
        auc = "-"
    else:
        auc = roc_auc_score(targets, prob_preds, average = "weighted", multi_class = "ovo") # multi_class = "ovr"
    precision = precision_score(targets, preds, average='weighted') #score-All average
    recall = recall_score(targets, preds, average='weighted') #score-All average
    f1 = f1_score(targets, preds, average='weighted') #score-All average
        
    classificationReport = classification_report(targets, preds, labels = labels, target_names = class_names, digits=5)

    logger.log(f"auc = {auc}")
    logger.log(f"accuracy = {accuracy}")
    logger.log(f"precision = {precision}")
    logger.log(f"recall = {recall}")
    logger.log(f"f1 = {f1}")
    logger.log(f"confusionMatrix = \n {confusionMatrix}")
    logger.log(f"classificationReport = \n {classificationReport}")


    results_dict = {}
    results_dict["auc"] = auc
    results_dict["accuracy"] = accuracy
    results_dict["precision"] = precision
    results_dict["recall"] = recall
    results_dict["f1"] = f1
    results_dict["confusionMatrix"] = confusionMatrix.tolist()
    results_dict["classificationReport"] = classificationReport

    return results_dict


def upsampleFeatures(labels, features):

    classes, count = np.unique(labels, return_counts = True)
    
    max_count = max(count)

    label_indices = []
    for c in classes:

        c_idx = np.where(labels == c)[0]
        assert np.unique(labels[c_idx]) == c, "Error! Wrong class index filtered."

        #Bug-GRG : Since we sample randomly some of the videos are never sampled/included. 
        # So, make sure to only sample additional required videos after including all videos at least once!
        #For the max count class, set replace to False as setting it True might exclude some samples from training
        # upsample_c_idx = np.random.choice(c_idx, size = max_count, replace = len(c_idx) < max_count)
        if len(c_idx) < max_count:
            # upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = len(c_idx) < max_count).tolist())
            upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = max_count > 2*len(c_idx)).tolist())
        else:
            upsample_c_idx = c_idx
        
        np.random.shuffle(upsample_c_idx)
        
        assert c_idx.shape == np.unique(upsample_c_idx).shape, "Error! Some videos where excluded on updampling."

        label_indices.extend(upsample_c_idx)

    assert len(label_indices) == max_count * len(classes)

    upsample_label_indices = label_indices

    upsampled_features = features[label_indices, :]
    upsampled_labels = labels[label_indices]

    classes, count = np.unique(upsampled_labels, return_counts = True)

    assert np.array_equal(count, max_count * np.ones(len(classes))), "Error! Upsampling didn't result in class-balance"

    return upsampled_labels, upsampled_features, upsample_label_indices



def fitMLmodels(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list, train_folds, test_folds,
                    exp_name, exp_path, prefix, task, reports_path, label_features, label_names, lung_severity_scores, test_label_dict, 
                    gt_label_dict, label_dict_path, BiomarkerModel, onlyEnd2End, UsePerPatientFeatures, causal_features, USE_CAUSAL_FEATURES):

    ##Fit ML model
    if onlyEnd2End:
        ml_models = {}
        ml_models["EndToEnd"] = end2endModelScores
    else:
        ml_models = {
                        "Decision Tree": ml.fitDecisionTree, 
                        "SVM": ml.fitSVM,
                        "Random Forest": ml.fitRandomForest,
                        "AdaBoost": ml.fitAdaBoost, 
                        "Nearest Neighbours": ml.fitNearestNeighbours, 
                        "MLP": ml.fitMLP,
                        "MLP Large": ml.fitLargeMLP,
                        # "DSR": dso.fitDSR,
                        # "DSO": dso.fitDSO,
                    }
    
        #Added EndToEnd classifier for end-to-end model
        if (BiomarkerModel or (not BiomarkerModel and prefix == "")) and not UsePerPatientFeatures:
            ml_models["EndToEnd"] = end2endModelScores

    #Fit model 
    report_path = os.path.join(reports_path, f"classification_report_{task}.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Classification report")

    logger.log(f"Exp name: {exp_name}")
    logger.log(f"Task: {task}")
    logger.log(f"Train folds: {train_folds}")
    logger.log(f"Test folds: {test_folds}")
    

    logger.log(f"\n\nVarious ML model acc for {task} Task\n\n")


    results_dict = {}

    for model_name, modelFunc in ml_models.items():

        logger.log(f"\n    *** Model {model_name} ***  \n")
            
        if model_name == "EndToEnd":
            
            if BiomarkerModel:
                model_results_dict, accuracy = biomarkerModelScores(exp_name, exp_path, lung_severity_scores, test_label_ft, gt_label_dict, test_video_list, label_features, label_names, task, logger, causal_features, USE_CAUSAL_FEATURES)
            else:
                model_results_dict, accuracy = end2endModelScores(exp_name, exp_path, lung_severity_scores, task, logger)
        
            logger.log(f"{model_name} acc = {accuracy}")

        else:

            model, accuracy, ml_predictions, ml_prob_predictions = modelFunc(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3)
            assert np.array_equal(ml_prob_predictions.argmax(1), ml_predictions) or model_name in ["SVM"], "Error! Prob predictions does not match predictions." #SVM Prob_preds isn't guranted to match [refer docs].

            logger.log(f"{model_name} acc = {accuracy}")

            model_results_dict = calScores(preds = ml_predictions, prob_preds = ml_prob_predictions, targets = gt_test_scores, class_names = lung_severity_scores, task = task, logger = logger)


        results_dict[model_name] = model_results_dict

        if model_name == "Decision Tree":
            decisonTreePostAnalysis(model, ml_predictions, accuracy, train_label_ft, test_label_ft, 
                    gt_train_scores, gt_test_scores, train_video_list, test_video_list, 
                    exp_name, task, reports_path, label_names, lung_severity_scores, test_label_dict, label_dict_path,
                    plotDecisionTree = False, BiomarkerModel = BiomarkerModel)

        #Save the model
        if model_name != "EndToEnd":
            dump(model, os.path.join(reports_path, f"ML_Model_{task}_{model_name.replace(' ', '-')}.joblib")) 

        # #Load the model
        # model = load('filename.joblib')


    # print(f"\n\nVarious ML model acc for {task} Task\n\n")

    # print(f"Decision Tree acc = {dtAcc}")
    # print(f"SVM acc = {svmAcc}")
    # print(f"Random Forest acc = {rfAcc}")
    # print(f"AdaBoost acc = {abAcc}")
    # print(f"Nearest Neighbours Classifier acc = {nnAcc}")
    # print(f"MLP Classifier acc = {mlpAcc}")
    # print(f"MLP Classifier 2 acc = {mlp2Acc}")

    utils.writeJson(results_dict, os.path.join(reports_path, f"classification_results_{task}.json"))

    logger.close()

    return results_dict


def getTrainAndTestFeatures(dataset_split_dict_path, exp_path, exp_name, label_dict_path, train_folds, test_folds, multi_task, use_probs, BiomarkerModel):


    if BiomarkerModel:
        
        generateModelFeaturePreds(dataset_split_dict_path, exp_path, exp_name, label_dict_path, train_folds, test_folds, multi_task, use_probs)

        test_label_dict = utils.readJson(label_dict_path)
        train_label_dict = utils.readJson(label_dict_path.replace("_test.json", "_train.json"))

    else:

        train_filenames = np.load(os.path.join(exp_path, exp_name, "Train_filenames.npy"))
        train_resnet_features = np.load(os.path.join(exp_path, exp_name, "Train_feature_preds.npy"))

        train_label_dict = {}
        for video, feature in zip(train_filenames, train_resnet_features):
            # train_label_dict[f"{video.split('-')[0]}/{video}.avi"] = feature 

            #Handle multi-clip
            if "pocovid" in video:
                dataset_split_dict = utils.readJson(dataset_split_dict_path)
                pocovid_videos = dataset_split_dict[f"fold{test_folds[0]}"][f"{test_folds[0]}_pt_videos"]
                video_name = [v for v in pocovid_videos if video in v][-1]
            else:
                video_name = f"{video.split('-')[0]}/{video}.avi"
            
            if video_name in train_label_dict:
                train_label_dict[video_name].append(feature)
            else:
                train_label_dict[video_name] = [feature] 
        
        for video_name, feature in train_label_dict.items():
            train_label_dict[video_name] = np.array(train_label_dict[video_name]).mean(axis = 0)
            # train_label_dict[video_name] = np.array(train_label_dict[video_name]).max(axis = 0)

        test_filenames = np.load(os.path.join(exp_path, exp_name, "Test_filenames.npy"))
        test_resnet_features = np.load(os.path.join(exp_path, exp_name, "Test_feature_preds.npy"))

        test_label_dict = {}
        for video, feature in zip(test_filenames, test_resnet_features):
            # test_label_dict[f"{video.split('-')[0]}/{video}.avi"] = feature

            #Handle multi-clip
            if "pocovid" in video:
                if "pocovid-Cov-Atlas-pleural" in video:
                    video
                dataset_split_dict = utils.readJson(dataset_split_dict_path)
                pocovid_videos = dataset_split_dict[f"fold{test_folds[0]}"][f"{test_folds[0]}_pt_videos"]
                video_name = [v for v in pocovid_videos if video in v][-1]
            else:
                video_name = f"{video.split('-')[0]}/{video}.avi"
            
            if video_name in test_label_dict:
                test_label_dict[video_name].append(feature)
            else:
                test_label_dict[video_name] = [feature] 
        
        for video_name, feature in test_label_dict.items():
            test_label_dict[video_name] = np.array(test_label_dict[video_name]).mean(axis = 0)
            # test_label_dict[video_name] = np.array(test_label_dict[video_name]).max(axis = 0)


    return train_label_dict, test_label_dict


import torch


def biomarkerModelScores(exp_name, exp_path, class_names, test_label_ft, gt_label_dict, test_video_list, label_features, label_names, task, logger, causal_features, USE_CAUSAL_FEATURES):


    gt_test_label_ft = np.array([np.hstack([gt_label_dict[video][f] for f in label_features]) for video in test_video_list])
    assert gt_test_label_ft.shape == (len(test_video_list), 38)


    '''
    #Causal featuers
    # a1, b0, b3, bo2, pt1, pt2, pl1, i2, pb2, e0, e2
    #  [1, 5, 8, 12, 14, 15, 18, 22, 27, 35, 37]
    '''
    # causal_features = [1, 5, 8, 12, 14, 15, 18, 22, 27, 35, 37]
    if USE_CAUSAL_FEATURES:
        gt_test_label_ft = gt_test_label_ft[:, causal_features]
        # label_names = np.array(label_names)[causal_features]

    pred_test_label_ft = test_label_ft > 0.5
    accuracy = (pred_test_label_ft == gt_test_label_ft).mean()
    print(f'ML model (Biomarker) accuracy = {accuracy}')

    # calScores(test_preds, test_prob_preds, test_targets, class_names, task, logger = None)
    results_dict = calScores(preds = pred_test_label_ft, prob_preds = test_label_ft, targets = gt_test_label_ft, class_names = label_names, task = task, logger = logger, binary_cross_entropy = True)

    return results_dict, accuracy


def end2endModelScores(exp_name, exp_path, class_names, task, logger):

    test_prob_preds = np.load(os.path.join(exp_path, exp_name, "Test_prob_preds.npy"))
    test_preds = np.load(os.path.join(exp_path, exp_name, "Test_preds.npy"))
    test_targets = np.load(os.path.join(exp_path, exp_name, "Test_targets.npy"))
    test_filenames = np.load(os.path.join(exp_path, exp_name, "Test_filenames.npy"))

    # if task == "diagnosis":
    #     assert np.array_equal(torch.sigmoid(torch.tensor(test_prob_preds)).numpy() > 0.5, test_preds), "Error! Prob predictions does not match predictions."
    # else:
    #     assert np.array_equal(test_prob_preds.argmax(1), test_preds), "Error! Prob predictions does not match predictions."

    assert np.array_equal(test_prob_preds.argmax(1), test_preds), "Error! Prob predictions does not match predictions."

    #Combine Multi-clip evaluation    
    video_filenames = np.unique(test_filenames)
        
    video_targets = []
    video_prob_preds = []
    for video in video_filenames:

        video_idx = np.where(np.array(test_filenames) == video)[0]
        
        target = test_targets[video_idx][0]
        prob_pred = test_prob_preds[video_idx].mean(0) #Take the mean across video multi-clips
        # prob_pred = test_prob_preds[video_idx].max(0) #Take the max across video multi-clips

        assert np.all(test_targets[video_idx] == target), f"Error! Target don't match for the same video {video} clips."
    
        video_targets.append(target)
        video_prob_preds.append(prob_pred)


    # #Convert probs to preds
    # if task == "diagnosis":
    #     # preds = torch.sigmoid(torch.tensor(prob_preds)).numpy() > 0.5
    #     video_prob_preds = torch.sigmoid(torch.tensor(video_prob_preds)).numpy()
    #     video_preds = video_prob_preds > 0.5
    # else:
    #     # preds = np.array(prob_preds).argmax(axis = 1)
    #     video_prob_preds = torch.softmax(torch.tensor(video_prob_preds), dim = 1).numpy()
    #     video_preds = video_prob_preds.argmax(axis = 1)

    video_prob_preds = torch.softmax(torch.tensor(video_prob_preds), dim = 1).numpy()
    video_preds = video_prob_preds.argmax(axis = 1)


    accuracy = (video_preds == video_targets).mean()
    print(f'ML model (EndToEnd) accuracy = {accuracy}')

    # calScores(test_preds, test_prob_preds, test_targets, class_names, task, logger = None)
    results_dict = calScores(video_preds, video_prob_preds, video_targets, class_names, task, logger = logger)

    return results_dict, accuracy


def saveFiles(task, reports_path, train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, 
                    train_video_list, test_video_list, upsample_label_indices):

    np.save(os.path.join(reports_path, f"{task}_train_label_ft.npy"), train_label_ft)
    np.save(os.path.join(reports_path, f"{task}_test_label_ft.npy"), test_label_ft)
    np.save(os.path.join(reports_path, f"{task}_gt_train_scores.npy"), gt_train_scores)
    np.save(os.path.join(reports_path, f"{task}_gt_test_scores.npy"), gt_test_scores)
    np.save(os.path.join(reports_path, f"{task}_train_video_list.npy"), train_video_list)
    np.save(os.path.join(reports_path, f"{task}_test_video_list.npy"), test_video_list)
    np.save(os.path.join(reports_path, f"{task}_upsample_label_indices.npy"), upsample_label_indices)


def fitTaskMLmodels(task, dataset_split_dict, train_folds, test_folds, train_label_dict, test_label_dict, gt_label_dict, label_features, 
        exp_name, exp_path, prefix, reports_path, label_names, label_dict_path,
        SF_category_dict, Disease_category_dict, upsampleTrainFeatures, BiomarkerModel, onlyEnd2End, 
        UsePerPatientFeatures, video_to_patient_wise_info_mapping_dict, causal_features, USE_CAUSAL_FEATURES, USE_SYNTHETIC_DATA, synthetic_data_path):


    if task == "lung-severity":
        lung_severity_scores = ['score-0', 'score-1', 'score-2', 'score-3']
    elif task == "sf-ratio":
        lung_severity_scores = [ '> 430', '275-430', '180-275', '<180', ]
    elif task == "diagnosis":
        lung_severity_scores = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others', ]
    else:
        raise ValueError(f"Wrong task = {task}!")


    train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list = [None]*6
    if BiomarkerModel or not onlyEnd2End:
        train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list = getFeatures(dataset_split_dict, train_folds, test_folds, train_label_dict, test_label_dict, \
        gt_label_dict, label_features, task, SF_category_dict, Disease_category_dict, BiomarkerModel, UsePerPatientFeatures, video_to_patient_wise_info_mapping_dict)


        '''
        #Causal featuers
        # a1, b0, b3, bo2, pt1, pt2, pl1, i2, pb2, e0, e2
        #  [1, 5, 8, 12, 14, 15, 18, 22, 27, 35, 37]
        '''
        # causal_features = [1, 5, 8, 12, 14, 15, 18, 22, 27, 35, 37]
        if USE_CAUSAL_FEATURES:
            train_label_ft = train_label_ft[:, causal_features]
            test_label_ft = test_label_ft[:, causal_features]


        #Upsample train set for class balancing
        if upsampleTrainFeatures:
            gt_train_scores, train_label_ft, upsample_label_indices = upsampleFeatures(labels = gt_train_scores, features = train_label_ft) 


        if USE_SYNTHETIC_DATA:
            synthetic_data = np.load(synthetic_data_path)

            syn_feat = synthetic_data[:, :38]
            syn_labels = synthetic_data[:, 38:]

            syn_labels = syn_labels.argmax(1)

            train_label_ft = np.concatenate((train_label_ft, syn_feat), axis = 0)
            gt_train_scores = np.concatenate((gt_train_scores, syn_labels), axis = 0)

            # train_label_ft = syn_feat
            # gt_train_scores = syn_labels
            
    
    saveFiles(task, reports_path, train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list, upsample_label_indices)

    results_dict = fitMLmodels(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list, train_folds, test_folds,
                    exp_name, exp_path, prefix, task, reports_path, label_features, label_names, lung_severity_scores, test_label_dict, 
                    gt_label_dict, label_dict_path, BiomarkerModel, onlyEnd2End, UsePerPatientFeatures, causal_features, USE_CAUSAL_FEATURES)
   
    return results_dict


def fitFoldMLmodels(train_folds, test_folds, exp_path, exp_name, prefix, dataset_split_dict_path, dataset_split_dict,
            label_features, label_names, SF_category_dict, Disease_category_dict, gt_label_dict, tasks_list,
            upsampleTrainFeatures, multi_task, use_probs, BiomarkerModel, onlyEnd2End, useGTFeatures_forTrain, useGTFeatures_forTest, 
            UsePerPatientFeatures, patient_wise_info_dict, video_to_patient_wise_info_mapping_dict, 
            consolidated_patient_info_dict, consolidation_type, causal_features, USE_CAUSAL_FEATURES, USE_SYNTHETIC_DATA, Synthetic_data_path):

    #Create reports dir to store resutls
    reports_path = os.path.join(exp_path, exp_name, f"reports{prefix}")
    utils.createDirIfDoesntExists(reports_path)


    # label_dict_path = "/data1/datasets/LSU-Dataset/user_label_gautamgare_12.json"
    # label_dict_path = 'user_label_tsmNet_1.json'
    label_dict_path = f'user_label_{exp_name}_test.json'
    label_dict_path = os.path.join(reports_path, label_dict_path)

    train_label_dict, test_label_dict = None, None
    if not (useGTFeatures_forTrain and useGTFeatures_forTest) and (BiomarkerModel or not onlyEnd2End):
        train_label_dict, test_label_dict = getTrainAndTestFeatures(dataset_split_dict_path, exp_path, exp_name, label_dict_path, train_folds, test_folds, 
                multi_task, use_probs, BiomarkerModel)
    
    # #Correct video names
    # corrected_gt_label_dict = {}
    # for v, l in gt_label_dict.items():
    #     corrected_gt_label_dict[v.split('/')[-1]] = l
    # gt_label_dict = corrected_gt_label_dict

    if useGTFeatures_forTrain:
        train_label_dict = gt_label_dict
    if useGTFeatures_forTest:
        test_label_dict = gt_label_dict
    
    if UsePerPatientFeatures:

        '''
        Method that aggreates patient-scan_date wise video features and consolidates into a single features.
        consolidation_types:
            1) max = Take max of every feature
            2) mean = Take mean of every feature
            3) concat = Concatinate every feature  
        '''
        def featureConsolidation(video_label_dict, video_to_patient_wise_info_mapping_dict, consolidation_type = "max", ignore_missing = False, groundtruth_labels = False):

            ##Aggregate features
            pt_label_dict = {}

            for video, features in video_label_dict.items():
                
                if ignore_missing:
                    if video not in video_to_patient_wise_info_mapping_dict:
                        print(f"Ignoring missing video - {video}")
                        continue

                patient_scan_name = video_to_patient_wise_info_mapping_dict[video]

                if patient_scan_name in pt_label_dict:
                    consolidated_features = pt_label_dict[patient_scan_name]
                else:
                    if BiomarkerModel or groundtruth_labels:
                        consolidated_features = {}
                    else:
                        consolidated_features = []

                if BiomarkerModel or groundtruth_labels:
                    for f, v in features.items():
                        
                        if f in consolidated_features:
                            cf = consolidated_features[f]
                            cf.append(v)
                            consolidated_features[f] = cf
                        else:
                            consolidated_features[f] = [v]            
                else:

                    consolidated_features.append([features])   
                
                pt_label_dict[patient_scan_name] = consolidated_features

            
            ##Combine features 
            consolidated_pt_label_dict = {}
            for patient_scan_name, pt_features in pt_label_dict.items():
                
                if BiomarkerModel or groundtruth_labels:

                    consolidated_pt_features = {}
                    for feature_name, feature in pt_features.items():
                        
                        if feature_name == "unusual_findings":
                            continue

                        feature = np.array(feature)

                        if consolidation_type == "max":
                            com_feature = feature.max(axis = 0)

                            assert com_feature.shape[0] == feature.shape[1], "Error! Feature consolidation done incorrectly."
                        elif consolidation_type == "mean":
                            com_feature = feature.mean(axis = 0)

                            assert com_feature.shape[0] == feature.shape[1], "Error! Feature consolidation done incorrectly."
                        elif consolidation_type == "concat":
                            com_feature = np.hstack(feature)

                            assert com_feature.shape[0] == feature.shape[0] * feature.shape[1], "Error! Feature consolidation done incorrectly."
                        else:
                            raise Exception(f"Error! Unsupported feature consoldiation type = {consolidation_type}")
                        
                        consolidated_pt_features[feature_name] = com_feature

                else:
                    
                    pt_features = np.vstack(pt_features)

                    if consolidation_type == "max":
                        consolidated_pt_features = pt_features.max(axis = 0)

                        assert consolidated_pt_features.shape[0] == pt_features.shape[1], "Error! Feature consolidation done incorrectly."
                    elif consolidation_type == "mean":
                        consolidated_pt_features = pt_features.mean(axis = 0)

                        assert consolidated_pt_features.shape[0] == pt_features.shape[1], "Error! Feature consolidation done incorrectly."
                    elif consolidation_type == "concat":
                        consolidated_pt_features = np.hstack(pt_features)

                        assert consolidated_pt_features.shape[0] == pt_features.shape[0] * pt_features.shape[1], "Error! Feature consolidation done incorrectly."
                    else:
                        raise Exception(f"Error! Unsupported feature consoldiation type = {consolidation_type}")


                consolidated_pt_label_dict[patient_scan_name] = consolidated_pt_features
            
            assert len(consolidated_pt_label_dict.keys()) == len(pt_label_dict.keys()), "Error! Missing some consolidated features."

            return consolidated_pt_label_dict

        train_label_dict = featureConsolidation(train_label_dict, video_to_patient_wise_info_mapping_dict, consolidation_type, ignore_missing = useGTFeatures_forTrain)
        test_label_dict = featureConsolidation(test_label_dict, video_to_patient_wise_info_mapping_dict, consolidation_type, ignore_missing = useGTFeatures_forTest)
        gt_label_dict = featureConsolidation(gt_label_dict, video_to_patient_wise_info_mapping_dict, consolidation_type, ignore_missing = True, groundtruth_labels = True)

    results_dict = {}

    ### Task-specific Code ###
    for task in tasks_list: 

        task_results_dict = fitTaskMLmodels(task, dataset_split_dict, train_folds, test_folds, train_label_dict, test_label_dict, gt_label_dict, label_features, 
            exp_name, exp_path, prefix, reports_path, label_names, label_dict_path,
            SF_category_dict, Disease_category_dict, upsampleTrainFeatures, BiomarkerModel, onlyEnd2End, 
            UsePerPatientFeatures, video_to_patient_wise_info_mapping_dict, causal_features, USE_CAUSAL_FEATURES, USE_SYNTHETIC_DATA, Synthetic_data_path)

        results_dict[task] = task_results_dict
    
    utils.writeJson(results_dict, os.path.join(reports_path, f"classification_results_all_tasks.json"))

    return results_dict


def combineFoldResults(results_dict, exp_name, exp_path, tasks_list, prefix, BiomarkerModel, onlyEnd2End, UsePerPatientFeatures, float_precision = 2, percent_factor = 100):

    if onlyEnd2End:
        # ml_models = []
        ml_models = ["EndToEnd"]
    else:
        ml_models = ["Decision Tree", "SVM", "Random Forest", "AdaBoost", "Nearest Neighbours", "MLP", "MLP Large"]

        #Added EndToEnd classifier for end-to-end model
        if (BiomarkerModel or (not BiomarkerModel and prefix == "")) and not UsePerPatientFeatures:
            ml_models = ["EndToEnd"] + ml_models


    #Combine results 
    report_path = os.path.join(exp_path, f"{exp_name}{prefix}_combined_exp_report.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Combined Exp Report")

    logger.log(f"Exp name: {exp_name}")

    folds_list = list(results_dict.keys())
    logger.log(f"Folds : {folds_list}")

    
    ### Task-specific Code ###
    for task in tasks_list: 
        
        logger.log(f"\n{'*'*35}\n    *** Task {task} ***  \n{'*'*35}\n")

        task_print_str = ""

        for model_name in ml_models:


            logger.log(f"\n    *** Model {model_name} ***  \n")

            auc_list = []
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []

            for val_fold in folds_list:

                fold_results_dict = results_dict[val_fold]
                task_results_dict = fold_results_dict[task]
                model_results_dict = task_results_dict[model_name]
                
                auc_list.append(model_results_dict["auc"])
                accuracy_list.append(model_results_dict["accuracy"])
                precision_list.append(model_results_dict["precision"])
                recall_list.append(model_results_dict["recall"])
                f1_list.append(model_results_dict["f1"])

            # assert np.array_equal(accuracy_list, recall_list) or task == "diagnosis", "Error! Weighted mean Acc != Recall."
            if model_name == "EndToEnd" and BiomarkerModel:
                auc_list = accuracy_list
                accuracy_list = recall_list #Need to retain recall list

            assert np.array_equal(accuracy_list, recall_list), "Error! Weighted mean Acc != Recall."

            logger.log(f"auc = {np.mean(auc_list)} +/- {np.std(auc_list)}")
            logger.log(f"accuracy = {np.mean(accuracy_list)} +/- {np.std(accuracy_list)}")
            logger.log(f"precision = {np.mean(precision_list)} +/- {np.std(precision_list)}")
            logger.log(f"recall = {np.mean(recall_list)} +/- {np.std(recall_list)}")
            logger.log(f"f1 = {np.mean(f1_list)} +/- {np.std(f1_list)}")

              
            # # logger.log(f"\n\n### Latex format ####\n\n")

            # print_str = f"\n\nLatex Entry: \n\n & & &  "
            # print_str += f"{np.mean(auc_list)*percent_factor:.{float_precision}f} $\pm$ {np.std(auc_list)*percent_factor:.{float_precision}f}"
            # print_str += f" & {np.mean(acc_list)*percent_factor:.{float_precision}f} $\pm$ {np.std(acc_list)*percent_factor:.{float_precision}f}"
            # print_str += f" & {np.mean(precision_list)*percent_factor:.{float_precision}f} $\pm$ {np.std(precision_list)*percent_factor:.{float_precision}f}"
            # print_str += f" & {np.mean(recall_list)*percent_factor:.{float_precision}f} $\pm$ {np.std(recall_list)*percent_factor:.{float_precision}f}"
            # print_str += f" & {np.mean(f1_list)*percent_factor:.{float_precision}f} $\pm$ {np.std(f1_list)*percent_factor:.{float_precision}f}"
            # print_str += f" \\\\ \n\n"
            # logger.log(print_str)

            # # logger.log(f"\n\nLatex Entry: \n\n & & &  {np.mean(score3_precision):.4f} $\pm$ {np.std(score3_precision):.4f} & {np.mean(score3_recall):.4f} $\pm$ {np.std(score3_recall):.4f} & {np.mean(score3_f1):.4f} $\pm$ {np.std(score3_f1):.4f} \\\\ \n\n")
            # # logger.log(f"\n\n### Latex format ####\n\n")

            def latexMeanStdFormat(metric):
                return f" & {np.mean(metric)*percent_factor:.{float_precision}f} $\pm$ {np.std(metric)*percent_factor:.{float_precision}f}"
          
            print_str = f"\n\nLatex Entry: \n\n {model_name} & "
            print_str += latexMeanStdFormat(auc_list)
            print_str += latexMeanStdFormat(accuracy_list)
            print_str += latexMeanStdFormat(precision_list)
            # print_str += latexMeanStdFormat(recall_list)
            print_str += latexMeanStdFormat(f1_list)
            print_str += f" &  \\\\ \n\n"
            logger.log(print_str)

            task_print_str += print_str 
        

        logger.log(f"\n   *** All Latex entries of {task} task ***   \n")
        logger.log(task_print_str)


    logger.close()



if __name__ == '__main__':

             

    datadir = "/data1/datasets/LSU-LargeV2-Dataset/"

    dataset_split_dict_path = os.path.join(datadir, "dataset_split_equi_class_R1.json")
    # dataset_split_dict_path = os.path.join(datadir, "dataset_split_equi_class_R2.json")
    dataset_split_dict = utils.readJson(dataset_split_dict_path)

    SF_category_dict = utils.readJson(path = os.path.join(datadir, "SF_category_dict.json"))
    Disease_category_dict = utils.readJson(path = os.path.join(datadir, "Disease_category_dict.json"))

    gt_label_dict = utils.readJson(os.path.join(datadir, "user_label_ggare_2.json"))
 
    patient_wise_info_dict_path = "/data1/datasets/LSU-LargeV2-Dataset/patient_wise_info_dict.json"
    patient_wise_info_dict = utils.readJson(patient_wise_info_dict_path)


    exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps"
    # exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/Diagnostic_Rules_Exps"
    # exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_ClinicalStudy_Exps"
    
      
    # exp_name = "GroundTruth_BiomarkerFeatures"

    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1"
    # exp_name = "tsm_LSU-Large-Dataset_Bio_Crop_upsampleVal_T1"
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2"
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2copy"
    # exp_name = "i3d_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_I3D_T1"

    #POCOVID
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2pocovid"
    
    #Abalation Study
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_NoSTAug_T1"
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_RN34_T1"

    #2Fold models
    # exp_name = "tsm_LSU-Large-Dataset_BioM_2Fold_Savg_EquiTS_Crop_T1"

    #2Fold models
    # train_folds = ["A"]
    # exp_name = "tsm_LSU-Large-Dataset_BioM_2Fold_Savg_EquiTS_Crop_T1_C"
    #End-to-end models
    # exp_name = "tsm_LSU-Large-Dataset_End2end_2Fold_Savg_EquiTS_Crop_T1_C"

    #End-to-end models

    # exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T1"
    # exp_name = "tsm_LSU-Large-Dataset_SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2"
    
    #POCOVID
    # exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2pocovid"
    
    #Abalation Study
    # exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_NoSTAug_T1"
    # exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_RN34_T1"

    #2Fold models
    # exp_name = "tsm_LSU-Large-Dataset_End2end_2Fold_Savg_EquiTS_Crop_T1"
    # exp_name = "tsm_LSU-Large-Dataset_End2endSF_2Fold_Savg_EquiTS_Crop_T2"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_2Fold_Savg_EquiTS_Crop_T2"

    # exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T1_C"
    # exp_name = "tsm_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_T1_C"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T1_C"

    # exp_name = "i3d_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_I3D_T1"
    # exp_name = "i3d_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_I3D_T2"
    # exp_name = "i3d_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_I3D_T2"

    #MICCAI-22 Final Models
    exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T2"
    # exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2"

    # prefix = ""
    # prefix = "_pre_patient_stats"
    # prefix = "_pre_patient_stats_max"
    # prefix = "_pre_patient_stats_max_v1"
    # prefix = "_pre_patient_stats_max_v2"
    # prefix = "_pre_patient_stats_max_v3"
    # prefix = "_pre_patient_stats_max_v4"
    # prefix = "_pre_patient_stats_max_v5"
    # prefix = "_pre_patient_stats_max_v6" #Best-results
    # prefix = "_pre_patient_stats_max_v7"
    # prefix = "_pre_patient_stats_mean"
    # prefix = "_pre_patient_stats_concat"
    # prefix = "causal_features"
    # prefix = "_pre_patient_max_causal_features_Upsampled_Acyclic_Disease_GT_TrainAB"
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_GT_TrainAB"
    # prefix = "_pre_patient_mean_causal_features_trial"
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB"
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB_trial" #Good result
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB_trial1"
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB_trial2"
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB_trial3"
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB_trial4"
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB_trial5"
    prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB_synthetic_data"
    # prefix = "_pre_patient_mean_causal_features_Upsampled_Acyclic_Disease_ModelContinue_TrainAB_only_synthetic_data"
    # prefix = "_ml_tasks"
    # prefix = "_ml_gt_tasks"
    # prefix = "_dso"
    # prefix = "_dso_tmp"
    # prefix="_end2end"
    # prefix = "_C"
    # prefix = "_C_end2end"
    # prefix = "_C_noUpsample"
    # prefix = "_t1"
    # prefix = "_t2"
    # prefix = "_t3"
    # prefix = "_t6" #"_t5" #"_t4"
    # prefix = "_diagnosis"
    # prefix = "_diagnosis_t3" #"_diagnosis_t2" #"_diagnosis_t1"
    # prefix = "_sf-ratio"
    # prefix = "_sf-ratio_end2end"
    

    UsePerPatientFeatures = True
    consolidation_type = "max" #"max", "mean", "concat"

    if UsePerPatientFeatures:

        video_to_patient_wise_info_mapping_dict = {}
        
        consolidated_patient_info_dict = {}
        pt_SF_category_dict = {}
        pt_Disease_category_dict = {}

        for patient, scan_features in patient_wise_info_dict.items():

            for scan_date, features in scan_features.items():

                patient_scan_name = f"{patient}/{scan_date}" 

                videos = features['videos']

                for video in videos:
                    video_name = f"{video.split('-')[0]}/{video}.avi"
                    video_to_patient_wise_info_mapping_dict[video_name] = patient_scan_name

                consolidated_patient_info_dict[patient_scan_name] = {'sf_cat': features['sf_cat'], 'disease_cat': features['disease_cat']}
                
                pt_SF_category_dict[patient_scan_name] = features['sf_cat']
                pt_Disease_category_dict[patient_scan_name] = features['disease_cat']
    
        #Update exisitng video-wise dict with patient-wise dict
        SF_category_dict = pt_SF_category_dict
        Disease_category_dict = pt_Disease_category_dict


    onlyEnd2End = False
    
    BiomarkerModel = True #True

    MULTI_TASK = False
    useProbs = True
    upsampleTrainFeatures = True #True

    useGTFeatures_forTrain = False
    useGTFeatures_forTest = False


    # tasks_list = ["lung-severity", "sf-ratio", "diagnosis"]
    # tasks_list = ["sf-ratio", "diagnosis"]
    # tasks_list = ["lung-severity"]
    # tasks_list = ["sf-ratio"]
    tasks_list = ["diagnosis"]


    label_names = [ 
                    'al-none', 'al-weak', 'al-bold', 'al-*also* stacked', 'al-*also* wide (> 2cm)',
                    'bl-none', 'bl-few (1-3)', 'bl-some (4-6)', 'bl-many|coalescing', "bl-\"white\" (no striations)",   
                    'bo-N/A', 'bo-pleura', 'bo-sub-plu', 
                    'pt-<1mm', 'pt-2-3mm', 'pt-4-5mm', 'pt->5mm',
                    'pl-top', 'pl-mid', 'pl-btm', 
                    'pi-none', 'pi-<5mm (few)', 'pi-<5mm (multiple)', 'pi-5-10mm', 'pi->10mm', 
                    'pb-none', 'pb-<5mm (few)', 'pb-<5mm (multiple)', 'pb-5-10mm', 'pb->10mm',
                    'cn-none', 'cn-<5mm (few)', 'cn-<5mm (multiple)', 'cn-5-10mm', 'cn->10mm',
                    'ef-none', 'ef-<5mm', 'ef->5mm', 
                ]

    label_features = ['alines', 'blines', 'blines_origin', 'pleural_thickness', 'pleural_location', 'pleural_indent', 'pleural_break', 'consolidation', 'effusion', ]

    # lung_severity_scores = ['score-0', 'score-1', 'score-2', 'score-3']

    '''
    #Causal featuers
    # a1, b0, b3, bo2, pt1, pt2, pl1, i2, pb2, e0, e2
    #  [1, 5, 8, 12, 14, 15, 18, 22, 27, 35, 37] #Model preds non-Acyclic Train(A,B)
    '''
    
    ##Severity prediction
    # causal_features_names = ['a0', 'a1', 'b0', 'b1', 'b2', 'b3', 'c0', 'c2', 'c3', 'e0', 'e2', 'i2', 'pb2', 'pl1', 'pt1', 'pt2'] #GroundTruth Train(A,B) Test(D)
    # causal_features_names = ['b0', 'b1', 'b2', 'b3', 'bo1', 'c0', 'c2', 'c3', 'e0', 'e2', 'i2', 'pb2', 'pb3', 'pl1', 'pt2'] #GroundTruth Acyclic Train(A,B) Test(D)
    # causal_features_names = ['b0', 'b1', 'b2', 'b3', 'bo1', 'c3', 'e0', 'i2', 'pb2', 'pb3', 'pl1', 'pl2', 'pt2'] #GroundTruth Acyclic Train(A,B,C) Test(D)
    
    ##Disease prediction
    # causal_features_names = ['a0', 'a1', 'a2', 'b0', 'b2', 'c0', 'e0', 'i2', 'pb1', 'pb2', 'pl1'] #GroundTruth Disease Acyclic Train(A,B,C) Test(D)
    # causal_features_names = ['a1', 'b3', 'c0', 'e0', 'i2', 'pb1', 'pl0', 'pl1', 'pt1', 'pt2'] #GroundTruth Disease Acyclic Train(A,B) Test(D)
    # causal_features_names = ['a1', 'a2', 'b0', 'b1', 'b3', 'bo1', 'c0', 'i1', 'pb1', 'pl0', 'pt1'] #Patient-max Upsampled GroundTruth Disease Acyclic Train(A,B) Test(D)
    # causal_features_names = ['a0', 'a1', 'a2', 'b0', 'bo1', 'c0', 'e0', 'i2', 'pl1', 'pt1', 'pt2'] #Patient-mean Upsampled GroundTruth Disease Acyclic Train(A,B) Test(D)
    # causal_features_names = ['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3', 'bo0', 'bo1', 'bo2', 'c0', 'c1', 'e0', 'i1', 'i2', 'pb0', 'pb1', 'pb2', 'pl1', 'pt1', 'pt2'] #Patient-mean Upsampled GroundTruth Disease Acyclic Train(A,B) Test(D) - sparsity 0.0004
    # causal_features_names = ['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3', 'bo0', 'bo1', 'bo2', 'c0', 'c1', 'c2', 'c3', 'c4', 'e0', 'i0', 'i1', 'i2', 'pb0', 'pb1', 'pb2', 'pl1', 'pt1', 'pt2', 'pt3'] #Trial
    # causal_features_names = ['a3', 'b0', 'b1', 'bo0', 'bo1', 'bo2', 'c0', 'i2', 'i4', 'pb2', 'pb4', 'pl1', 'pt0', 'pt1', 'pt3'] #Patient-max Upsampled Model Continous 
    # causal_features_names =['a0', 'a2', 'a3', 'a4', 'b0', 'b1', 'b3', 'b4', 'bo0', 'bo1', 'bo2', 'c0', 'c2', 'c3', 'c4', 'e0', 'i0', 'i1', 'i2', 'i4', 'pb0', 'pb1', 'pb2', 'pb4', 'pl0', 'pl1', 'pt0', 'pt1', 'pt2', 'pt3'] #Patient-max Upsampled Model Continous - Trial
    # causal_features_names = ['a2', 'a3', 'a4', 'b0', 'b1', 'bo0', 'c0', 'c4', 'e0', 'i0', 'i1', 'i2', 'pb0', 'pb1', 'pb2', 'pb4', 'pl0', 'pl1', 'pt0', 'pt1']
    # causal_features_names =  ['a0', 'a2', 'a3', 'a4', 'b0', 'b1', 'b4', 'bo0', 'bo1', 'bo2', 'c0', 'c1', 'c3', 'c4', 'e0', 'e1', 'e2', 'i0', 'i1', 'i2', 'i3', 'i4', 'pb0', 'pb1', 'pb4', 'pl0', 'pl1', 'pl2', 'pt0', 'pt1', 'pt2', 'pt3']
    # causal_features_names = ['a3', 'a4', 'b0', 'b1', 'bo0', 'c0', 'i0', 'i1', 'pb0', 'pb1', 'pl0', 'pl1', 'pt0']
    # causal_features_names = ['a3', 'b0', 'b1', 'b2', 'bo0', 'bo2', 'c0', 'c4', 'e0', 'i2', 'pb0', 'pb2', 'pb4', 'pl0', 'pl1', 'pt0']
    # causal_features_names = ['a0', 'a2', 'a3', 'a4', 'b0', 'b1', 'bo0', 'c0', 'c1', 'c4', 'e1', 'i2', 'pb0', 'pb1', 'pb2', 'pb4', 'pl0', 'pl1', 'pt0', 'pt1', 'pt3']
    causal_features_names = ['a0', 'a2', 'a3', 'a4', 'b0', 'b1', 'b3', 'b4', 'bo0', 'bo1', 'bo2', 'c0', 'c2', 'c3', 'c4', 'e0', 'i0', 'i1', 'i2', 'i4', 'pb0', 'pb1', 'pb2', 'pb4', 'pl0', 'pl1', 'pt0', 'pt1', 'pt2', 'pt3']

    USE_CAUSAL_FEATURES = False
    

    #Relabel nodes
    causal_feature_mapping = {
        "a0":0, "a1":1, "a2":2, "a3":3, "a4":4,
        "b0":5, "b1":6, "b2":7, "b3":8, "b4":9,
        "bo0":10, "bo1":11, "bo2":12,
        "pt0":13, "pt1":14, "pt2":15, "pt3":16,
        "pl0":17, "pl1":18, "pl2":19,
        "i0":20, "i1":21, "i2":22, "i3":23, "i4":24,
        "pb0":25, "pb1":26, "pb2":27, "pb3":28, "pb4":29,
        "c0":30, "c1":31, "c2":32, "c3":33, "c4":34,
        "e0":35, "e1":36, "e2":37,
        # "s0":38, "s1":39, "s2":40, "s3":41
    }

    causal_features = [causal_feature_mapping[f] for f in causal_features_names]

    if USE_CAUSAL_FEATURES:
        label_names = np.array(label_names)[causal_features].tolist()

    USE_SYNTHETIC_DATA = True
    # Synthetic_data_path = "/home/grg/Research/ENCO/checkpoints/2022_26_Acyclic_ModelBiomarkers-max_TrainUpsampledPatient/generateContinousSamples_500 (best)/synthetic_data_500.npz.npy"
    # Synthetic_data_path = "/home/grg/Research/ENCO/checkpoints/2022_26_Acyclic_ModelBiomarkers-max_TrainUpsampledPatient/generateContinousSamples_200 (best)/synthetic_data_200.npy"
    # Synthetic_data_path = "/home/grg/Research/ENCO/checkpoints/2022_26_Acyclic_ModelBiomarkers-max_TrainUpsampledPatient/generateContinousSamples_100 (best)/synthetic_data_100.npy"
    # Synthetic_data_path = "/home/grg/Research/ENCO/checkpoints/2022_26_Acyclic_ModelBiomarkers-max_TrainUpsampledPatient/generateContinousSamples_100/synthetic_data_100.npy"
    Synthetic_data_path = "/home/grg/Research/ENCO/checkpoints/2022_26_Acyclic_ModelBiomarkers-max_TrainUpsampledPatient/generateContinousSamples_50/synthetic_data_50.npy"
    
    ### Fold-specific Code ###
    
    folds_list = ["A", "B", "C"]
    # folds_list = ["A", "C"] #2-folds
    # folds_list = ["C"] #Abalation-study
    
    results_dict = {}
    
    for val_fold in folds_list:

        # train_folds = ["A", "B"]
        # train_folds = ["B", "C"]
        # train_folds = ["A", "C"]
        train_folds = [f for f in folds_list if f not in val_fold]

        test_folds = ["D"]
        # test_folds = ["E"] #pocovid

        fold_exp_name = f"{exp_name}_{val_fold}"

        print(f"Fitting models on {fold_exp_name}")
        print(f"Train folds: {train_folds}")
        print(f"Test folds: {test_folds}")

        fold_results_dict = fitFoldMLmodels(
                train_folds, test_folds, exp_path, fold_exp_name, prefix, dataset_split_dict_path, dataset_split_dict,
                label_features, label_names, SF_category_dict, Disease_category_dict, gt_label_dict, tasks_list,
                upsampleTrainFeatures, multi_task = MULTI_TASK, use_probs = useProbs, 
                BiomarkerModel = BiomarkerModel, onlyEnd2End = onlyEnd2End,
                useGTFeatures_forTrain = useGTFeatures_forTrain, useGTFeatures_forTest = useGTFeatures_forTest,
                UsePerPatientFeatures = UsePerPatientFeatures, patient_wise_info_dict = patient_wise_info_dict,
                video_to_patient_wise_info_mapping_dict = video_to_patient_wise_info_mapping_dict, 
                consolidated_patient_info_dict = consolidated_patient_info_dict, consolidation_type = consolidation_type, 
                causal_features = causal_features, USE_CAUSAL_FEATURES = USE_CAUSAL_FEATURES,
                USE_SYNTHETIC_DATA = USE_SYNTHETIC_DATA, Synthetic_data_path = Synthetic_data_path
            )


        results_dict[val_fold] = fold_results_dict
    
    utils.writeJson(results_dict, os.path.join(exp_path, f"{exp_name}{prefix}_classification_results.json"))

    # combineFoldResults(results_dict, folds_list, exp_name, exp_path, tasks_list, prefix, BiomarkerModel, onlyEnd2End)
    combineFoldResults(results_dict, exp_name, exp_path, tasks_list, prefix, BiomarkerModel, onlyEnd2End, UsePerPatientFeatures)

    print("finished!")
