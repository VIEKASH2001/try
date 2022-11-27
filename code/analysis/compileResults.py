import numpy as np
import torch

import os


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, multilabel_confusion_matrix



if __name__ == "__main__":

    exp_dir = "/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_PlMl_T1_C"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_PlMl_T1_A"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_PlMl_T1_B"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T1_C"
    exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T1_A"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T1_B"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_2Fold_Savg_EquiTS_Crop_T1_C"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_2Fold_Savg_EquiTS_Crop_T1_A"

    test_prob_preds = np.load(os.path.join(exp_dir, exp_name, "Test_prob_preds.npy"))
    test_targets = np.load(os.path.join(exp_dir, exp_name, "Test_targets.npy"))
    test_filenames = np.load(os.path.join(exp_dir, exp_name, "Test_filenames.npy"))

    test_preds = torch.sigmoid(torch.tensor(test_prob_preds)).numpy() > 0.5

    acc = (test_preds == test_targets).mean()
    print(f"Accuracy = {acc}")

    class_names = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others', ]
    labels = np.arange(len(class_names))

    accuracy = accuracy_score(test_targets, test_preds)

   
    confusionMatrix = multilabel_confusion_matrix(test_targets, test_preds, labels = labels)
    
    # confusionMatrix = confusion_matrix(targets, preds, labels = labels)
        
    classificationReport = classification_report(test_targets, test_preds, labels = labels, target_names = class_names, digits=5)

    print(f"multi-label accuracy = {accuracy}")
    print(f"confusionMatrix = \n {confusionMatrix}")
    print(f"classificationReport = \n {classificationReport}")


    pass


