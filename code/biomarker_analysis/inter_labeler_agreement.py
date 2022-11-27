import numpy as np

import torch 

import os 


import sys
sys.path.append("./code")
import utils


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, multilabel_confusion_matrix
from scipy.special import softmax
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from joblib import dump, load

# def calScores(preds, prob_preds, targets, class_names, task, logger):
def calScores(preds, targets, class_names, task, logger, prob_preds = None, task_print_str = None, float_precision = 2, percent_factor = 100):

    labels = np.arange(len(class_names))

    accuracy = accuracy_score(targets, preds)

    confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    auc = "-"
    if prob_preds is not None:
        auc = roc_auc_score(targets, prob_preds, average = "weighted", multi_class = "ovo") # multi_class = "ovr"
    precision = precision_score(targets, preds, average='weighted') #score-All average
    recall = recall_score(targets, preds, average='weighted') #score-All average
    f1 = f1_score(targets, preds, average='weighted') #score-All average
        
    classificationReport = classification_report(targets, preds, labels = labels, target_names = class_names, digits=5)

    assert np.array_equal(accuracy, recall), "Error! Weighted mean Acc != Recall."

    logger.log(f"auc = {auc}")
    logger.log(f"accuracy = {accuracy}")
    logger.log(f"precision = {precision}")
    logger.log(f"recall = {recall}")
    logger.log(f"f1 = {f1}")
    logger.log(f"confusionMatrix = \n {confusionMatrix}")
    logger.log(f"classificationReport = \n {classificationReport}")

    def latexMeanStdFormat(metric):
        return f" & {metric*percent_factor:.{float_precision}f}"
          
    print_str = f"\n\nLatex Entry: \n\n {task} "
    print_str += latexMeanStdFormat(auc) if auc != "-" else f" & {auc}"
    print_str += latexMeanStdFormat(accuracy)
    print_str += latexMeanStdFormat(precision)
    # print_str += latexMeanStdFormat(recall_list)
    print_str += latexMeanStdFormat(f1)
    print_str += f" \\\\ \n\n"
    logger.log(print_str)
    
    if task_print_str is not None:
        task_print_str += print_str 

    results_dict = {}
    results_dict["auc"] = auc
    results_dict["accuracy"] = accuracy
    results_dict["precision"] = precision
    results_dict["recall"] = recall
    results_dict["f1"] = f1
    results_dict["confusionMatrix"] = confusionMatrix.tolist()
    results_dict["classificationReport"] = classificationReport

    return results_dict, task_print_str



if __name__ == "__main__":

    ben_labels = utils.readJson("/data1/datasets/LSU-Large-Dataset/user_label_bdeboi_3.json")
    grg_labels = utils.readJson("/data1/datasets/LSU-Large-Dataset/user_label_ggare_2.json")
    tom_labels = utils.readJson("/data1/datasets/LSU-Large-Dataset/user_label_thf214_4.json")


    datadir = "/data1/datasets/LSU-Large-Dataset/"
    dataset_split_dict_path = os.path.join(datadir, "dataset_split_equi_class_R1.json")
    # dataset_split_dict_path = os.path.join(datadir, "dataset_split_equi_class_R2.json")
    dataset_split_dict = utils.readJson(dataset_split_dict_path)

    test_folds = ["D"]
    
    exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps"
    # exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_ClinicalStudy_Exps"
    reports_path = exp_path

    lung_severity_scores = ['score-0', 'score-1', 'score-2', 'score-3']


    test_video_list = []
    [test_video_list.extend(dataset_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in test_folds]

    # #Exclude videos
    # test_video_list = [v for v in test_video_list if "166-b" not in v and "160-7" not in v and "76-7" not in v and "76-8" not in v]

    ben_testset_severity_scores = [ben_labels[v]['lung-severity'] for v in test_video_list]
    ben_testset_severity_scores = np.array(ben_testset_severity_scores).argmax(1)
    assert ben_testset_severity_scores.shape[0] == len(test_video_list), "Error! Missing test video labels."
    assert np.array_equal(np.unique(ben_testset_severity_scores), np.arange(len(lung_severity_scores))), "Error! Label classes missing."

    grg_testset_severity_scores = [grg_labels[v]['lung-severity'] for v in test_video_list]
    grg_testset_severity_scores = np.array(grg_testset_severity_scores).argmax(1)
    assert grg_testset_severity_scores.shape[0] == len(test_video_list), "Error! Missing test video labels."
    assert np.array_equal(np.unique(grg_testset_severity_scores), np.arange(len(lung_severity_scores))), "Error! Label classes missing."

    tom_testset_severity_scores = [tom_labels[v]['lung-severity'] for v in test_video_list]
    tom_testset_severity_scores = np.array(tom_testset_severity_scores).argmax(1)
    assert tom_testset_severity_scores.shape[0] == len(test_video_list), "Error! Missing test video labels."
    assert np.array_equal(np.unique(tom_testset_severity_scores), np.arange(len(lung_severity_scores))), "Error! Label classes missing."

    logger = utils.Logger(os.path.join(reports_path, "inter_labeler_agreement_report_v1.txt"))
    task_print_str = ""

    logger.log(f"Inter Labeler Agreement report")

    # logger.log(f"Exp name: {exp_name}")
    # logger.log(f"Task: {task}")
    # logger.log(f"Train folds: {train_folds}")
    logger.log(f"Test folds: {test_folds}")
    

    task = "GRG as pred and Dr.Ben as ground-truth"

    logger.log(f"\n\nInter-labeler agreement between {task} \n\n")


    results_dict, task_print_str = calScores(preds = grg_testset_severity_scores, targets = ben_testset_severity_scores, class_names = lung_severity_scores, task = task, task_print_str = task_print_str, logger = logger)

    utils.writeJson(results_dict, os.path.join(reports_path, f"inter_labeler_agreement_results_{task.replace(' ', '-')}.json"))


    task = "Dr. Tom as pred and Dr.Ben as ground-truth"

    logger.log(f"\n\nInter-labeler agreement between {task} \n\n")


    results_dict, task_print_str = calScores(preds = tom_testset_severity_scores, targets = ben_testset_severity_scores, class_names = lung_severity_scores, task = task, task_print_str = task_print_str, logger = logger)

    utils.writeJson(results_dict, os.path.join(reports_path, f"inter_labeler_agreement_results_{task.replace(' ', '-')}.json"))


    task = "GRG as pred and Dr.Tom as ground-truth"

    logger.log(f"\n\nInter-labeler agreement between {task} \n\n")


    results_dict, task_print_str = calScores(preds = grg_testset_severity_scores, targets = tom_testset_severity_scores, class_names = lung_severity_scores, task = task, task_print_str = task_print_str, logger = logger)

    utils.writeJson(results_dict, os.path.join(reports_path, f"inter_labeler_agreement_results_{task.replace(' ', '-')}.json"))



    ### End-to-end model ###

    # e2e_exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2_C"
    # e2e_exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2_B"
    e2e_exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2_A"
    # e2e_exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T1_A"

    e2e_test_prob_preds = np.load(os.path.join(exp_path, e2e_exp_name, "Test_prob_preds.npy"))
    e2e_test_preds = np.load(os.path.join(exp_path, e2e_exp_name, "Test_preds.npy"))
    e2e_test_targets = np.load(os.path.join(exp_path, e2e_exp_name, "Test_targets.npy"))
    e2e_test_filenames = np.load(os.path.join(exp_path, e2e_exp_name, "Test_filenames.npy"))


    e2e_test_resnet_features = np.load(os.path.join(exp_path, e2e_exp_name, "Test_feature_preds.npy"))

    assert np.array_equal(e2e_test_prob_preds.argmax(1), e2e_test_preds), "Error! Prob predictions does not match predictions."

    #Combine Multi-clip evaluation    
    video_filenames = np.unique(e2e_test_filenames)
        
    video_targets = []
    video_prob_preds = []
    for video in video_filenames:

        video_idx = np.where(np.array(e2e_test_filenames) == video)[0]
        
        target = e2e_test_targets[video_idx][0]
        prob_pred = e2e_test_prob_preds[video_idx].mean(0) #Take the mean across video multi-clips
        # prob_pred = test_prob_preds[video_idx].max(0) #Take the max across video multi-clips

        assert np.all(e2e_test_targets[video_idx] == target), f"Error! Target don't match for the same video {video} clips."
    
        video_targets.append(target)
        video_prob_preds.append(prob_pred)

    video_prob_preds = torch.softmax(torch.tensor(video_prob_preds), dim = 1).numpy()
    video_preds = video_prob_preds.argmax(axis = 1)

    video_preds_dict, video_prob_preds_dict = {}, {}
    for video, video_pd, video_prob_pd in zip(video_filenames, video_preds, video_prob_preds):
        video_name = f"{video.split('-')[0]}/{video}.avi"
        video_preds_dict[video_name] = video_pd
        video_prob_preds_dict[video_name] = video_prob_pd

    
    e2e_testset_severity_scores = [video_preds_dict[v] for v in test_video_list]
    assert len(e2e_testset_severity_scores) == len(test_video_list), "Error! Missing test video labels."
    assert np.array_equal(np.unique(e2e_testset_severity_scores), np.arange(len(lung_severity_scores))), "Error! Label classes missing."
    e2e_testset_severity_prob_scores = [video_prob_preds_dict[v] for v in test_video_list]
    assert np.array_equal(np.array(e2e_testset_severity_prob_scores).argmax(1), e2e_testset_severity_scores), "Error! Prob preds don't match."

    task = "E2E as pred and Dr.Ben as ground-truth"

    logger.log(f"\n\nInter-labeler agreement between {task} \n\n")


    results_dict, task_print_str = calScores(preds = e2e_testset_severity_scores, targets = ben_testset_severity_scores, class_names = lung_severity_scores, task = task, 
                prob_preds = e2e_testset_severity_prob_scores, task_print_str = task_print_str, logger = logger)
    
    utils.writeJson(results_dict, os.path.join(reports_path, f"inter_labeler_agreement_results_{task.replace(' ', '-')}.json"))

    #Export preds to json
    preds_dict = {}
    for v, p in zip(test_video_list, e2e_testset_severity_scores):
        severity = [0, 0, 0, 0]
        severity[p] = 1
        preds_dict[v] = {"lung-severity": severity}
    utils.writeJson(preds_dict, os.path.join(reports_path, f"{task.split('as')[0][:-1].replace(' ', '-')}.json"))


    ### End-to-end model test-resnet-features ###


    e2e_test_label_dict = {}
    for video, feature in zip(e2e_test_filenames, e2e_test_resnet_features):
        # test_label_dict[f"{video.split('-')[0]}/{video}.avi"] = feature

        #Handle multi-clip
        video_name = f"{video.split('-')[0]}/{video}.avi"
        if video_name in e2e_test_label_dict:
            e2e_test_label_dict[video_name].append(feature)
        else:
            e2e_test_label_dict[video_name] = [feature] 
    
    for video_name, feature in e2e_test_label_dict.items():
        e2e_test_label_dict[video_name] = np.array(e2e_test_label_dict[video_name]).mean(axis = 0)
        # test_label_dict[video_name] = np.array(test_label_dict[video_name]).max(axis = 0)

    e2e_test_label_ft = np.array([e2e_test_label_dict[video] for video in test_video_list])
    assert e2e_test_label_ft.shape == (len(test_video_list), 512)


    ### End-to-end model SVM ###

    e2e_svm_model = load(os.path.join(exp_path, e2e_exp_name, "reports", "ML_Model_lung-severity_SVM.joblib"))
    e2e_svm_testset_severity_scores = e2e_svm_model.predict(e2e_test_label_ft)
    e2e_svm_testset_severity_prob_scores = e2e_svm_model.predict_proba(e2e_test_label_ft)
    assert len(e2e_svm_testset_severity_scores) == len(test_video_list), "Error! Missing test video labels."
    assert np.array_equal(np.unique(e2e_svm_testset_severity_scores), np.arange(len(lung_severity_scores))), "Error! Label classes missing."

    task = "E2E (SVM) as pred and Dr.Ben as ground-truth"

    logger.log(f"\n\nInter-labeler agreement between {task} \n\n")


    results_dict, task_print_str = calScores(preds = e2e_svm_testset_severity_scores, targets = ben_testset_severity_scores, class_names = lung_severity_scores, task = task, 
                prob_preds = e2e_svm_testset_severity_prob_scores, task_print_str = task_print_str, logger = logger)

    utils.writeJson(results_dict, os.path.join(reports_path, f"inter_labeler_agreement_results_{task.replace(' ', '-')}.json"))

    #Export preds to json
    preds_dict = {}
    for v, p in zip(test_video_list, e2e_svm_testset_severity_scores):
        severity = [0, 0, 0, 0]
        severity[p] = 1
        preds_dict[v] = {"lung-severity": severity}
    utils.writeJson(preds_dict, os.path.join(reports_path, f"{task.split('as')[0][:-1].replace(' ', '-')}.json"))


    ### End-to-end model MLP ###


    e2e_mlp_model = load(os.path.join(exp_path, e2e_exp_name, "reports", "ML_Model_lung-severity_MLP.joblib"))
    e2e_mlp_testset_severity_scores = e2e_mlp_model.predict(e2e_test_label_ft)
    e2e_mlp_testset_severity_prob_scores = e2e_mlp_model.predict_proba(e2e_test_label_ft)
    assert len(e2e_mlp_testset_severity_scores) == len(test_video_list), "Error! Missing test video labels."
    assert np.array_equal(np.unique(e2e_mlp_testset_severity_scores), np.arange(len(lung_severity_scores))), "Error! Label classes missing."

    task = "E2E (MLP) as pred and Dr.Ben as ground-truth"

    logger.log(f"\n\nInter-labeler agreement between {task} \n\n")


    results_dict, task_print_str = calScores(preds = e2e_mlp_testset_severity_scores, targets = ben_testset_severity_scores, class_names = lung_severity_scores, task = task, 
                prob_preds = e2e_mlp_testset_severity_prob_scores, task_print_str = task_print_str, logger = logger)

    utils.writeJson(results_dict, os.path.join(reports_path, f"inter_labeler_agreement_results_{task.replace(' ', '-')}.json"))

    #Export preds to json
    preds_dict = {}
    for v, p in zip(test_video_list, e2e_mlp_testset_severity_scores):
        severity = [0, 0, 0, 0]
        severity[p] = 1
        preds_dict[v] = {"lung-severity": severity}
    utils.writeJson(preds_dict, os.path.join(reports_path, f"{task.split('as')[0][:-1].replace(' ', '-')}.json"))



    ### Biomarker model test features ###

    # bio_exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_C"
    # bio_exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_B"
    bio_exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_A"
    # bio_exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1_A"
    
    label_features = ['alines', 'blines', 'blines_origin', 'pleural_thickness', 'pleural_location', 'pleural_indent', 'pleural_break', 'consolidation', 'effusion', ]

    bio_test_label_dict = utils.readJson(os.path.join(exp_path, bio_exp_name, "reports_t1", f"user_label_{bio_exp_name}_test.json"))
    bio_test_label_ft = np.array([np.hstack([bio_test_label_dict[video][f] for f in label_features]) for video in test_video_list])
    assert bio_test_label_ft.shape == (len(test_video_list), 38)


    ### Biomarker model SVM ###

    bio_svm_model = load(os.path.join(exp_path, bio_exp_name, "reports_t1", "ML_Model_lung-severity_SVM.joblib"))
    bio_svm_testset_severity_scores = bio_svm_model.predict(bio_test_label_ft)
    bio_svm_testset_severity_prob_scores = bio_svm_model.predict_proba(bio_test_label_ft)
    assert len(bio_svm_testset_severity_scores) == len(test_video_list), "Error! Missing test video labels."
    assert np.array_equal(np.unique(bio_svm_testset_severity_scores), np.arange(len(lung_severity_scores))), "Error! Label classes missing."

    task = "Bio (SVM) as pred and Dr.Ben as ground-truth"

    logger.log(f"\n\nInter-labeler agreement between {task} \n\n")


    results_dict, task_print_str = calScores(preds = bio_svm_testset_severity_scores, targets = ben_testset_severity_scores, class_names = lung_severity_scores, task = task, 
                prob_preds = bio_svm_testset_severity_prob_scores, task_print_str = task_print_str, logger = logger)

    utils.writeJson(results_dict, os.path.join(reports_path, f"inter_labeler_agreement_results_{task.replace(' ', '-')}.json"))

    #Export preds to json
    preds_dict = {}
    for v, p in zip(test_video_list, bio_svm_testset_severity_scores):
        severity = [0, 0, 0, 0]
        severity[p] = 1
        preds_dict[v] = {"lung-severity": severity}
    utils.writeJson(preds_dict, os.path.join(reports_path, f"{task.split('as')[0][:-1].replace(' ', '-')}.json"))


    ### Biomarker model MLP ###


    bio_mlp_model = load(os.path.join(exp_path, bio_exp_name, "reports_t1", "ML_Model_lung-severity_MLP.joblib"))
    bio_mlp_testset_severity_scores = bio_mlp_model.predict(bio_test_label_ft)
    bio_mlp_testset_severity_prob_scores = bio_mlp_model.predict_proba(bio_test_label_ft)
    assert len(bio_mlp_testset_severity_scores) == len(test_video_list), "Error! Missing test video labels."
    assert np.array_equal(np.unique(bio_mlp_testset_severity_scores), np.arange(len(lung_severity_scores))), "Error! Label classes missing."

    task = "Bio (MLP) as pred and Dr.Ben as ground-truth"

    logger.log(f"\n\nInter-labeler agreement between {task} \n\n")


    results_dict, task_print_str = calScores(preds = bio_mlp_testset_severity_scores, targets = ben_testset_severity_scores, class_names = lung_severity_scores, task = task, 
                prob_preds = bio_mlp_testset_severity_prob_scores, task_print_str = task_print_str, logger = logger)

    utils.writeJson(results_dict, os.path.join(reports_path, f"inter_labeler_agreement_results_{task.replace(' ', '-')}.json"))
    
    #Export preds to json
    preds_dict = {}
    for v, p in zip(test_video_list, bio_mlp_testset_severity_scores):
        severity = [0, 0, 0, 0]
        severity[p] = 1
        preds_dict[v] = {"lung-severity": severity}
    utils.writeJson(preds_dict, os.path.join(reports_path, f"{task.split('as')[0][:-1].replace(' ', '-')}.json"))



    logger.log(f"\n   *** All Latex entries of {task} task ***   \n")
    logger.log(task_print_str)

    #Close report file
    logger.close()
    
    print(f"Finished!")