from matplotlib import pyplot as plt
import numpy as np
import json

from sklearn import tree, svm

import os

# import sys
# sys.path.append("./code")
# from dataset.video_dataset import *

# import constants
# random_split_trial = 'R1'
    
# constants.dataset_path = os.path.join("data", "LSU-Lung-Severity-Dataset")
# constants.processed_dataset_path = os.path.join("data", "LSU-Lung-Severity-Dataset", "processed")
# # video_type = ''

# train_dataset = LSULungDataset(dataset_type = 'train', random_split_trial = random_split_trial, biomarkerLabels = True)


def loadPredLabels(y_prob_path, dataset_split_dict, dataset_type, trial):

    y_prob = np.load(y_prob_path)

    video_list = []
    if dataset_type == 'train':
        # [[video_list.extend(s) for s in f.values()] for f in dataset_split_dict[dataset_type].values()]
        # [[video_list.extend(s) for s in f.values()] for k,f in dataset_split_dict[dataset_type].items() if k != 'A']
        [[video_list.extend(s) for k, s in f.items() if k != trial] for f in dataset_split_dict[dataset_type].values()]
    else:
        [video_list.extend(s) for s in dataset_split_dict[dataset_type].values()]

    assert len(video_list) == y_prob.shape[0], "No of videos mismatch!"
    
    y_prob_pred = np.array([[int(p > 0.5) for p in l] for l in y_prob])

    label_dict = {}
    for idx, video in enumerate(video_list):

        # pred = y_prob[idx].astype(np.int)
        pred = y_prob_pred[idx]

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

        label_dict[video] = video_label
    
    return video_list, label_dict



if __name__ == '__main__':

    base_path = "/home/grg/OCT_Ultrasound_Segmentation/"

    trial = "E"

    # exp_name = "BME_Image_Biomarkers/tsm_LSU-Lung-Severity-Dataset_T3_biomarker_labels_A_R1"
    exp_name = f"BME_Image_Biomarkers/tsm_LSU-Lung-Severity-Dataset_T3a_biomarker_labels_{trial}_R1"

    train_y_prob_path = os.path.join(base_path, 'results', exp_name, 'Prob_based_Train_Clasification_y_prob.npy')

    test_y_prob_path = os.path.join(base_path, 'results', exp_name, 'Prob_based_Test_Clasification_y_prob.npy')
    
    dataset_split_dict_path = os.path.join(base_path, 'data/LSU-Lung-Severity-Dataset/dataset_split_equi_class_R1.json')
    
    # label_dict_path = 'user_label_tsmNet_5.json'
    
    gt_label_dict_path = os.path.join(base_path, 'data/LSU-Lung-Severity-Dataset/user_label_gautamgare_12.json')
    
    
    label_names = [ 
                    'al-none', 'al-weak', 'al-bold', 'al-*also* stacked', 'al-*also* wide (> 2cm)',
                    'bl-none', 'bl-few (1-3)', 'bl-some (4-5)', 'bl-many|coalescing', "bl-\"white\" (no striations)",   
                    'pi-none', 'pi-<5mm (single)', 'pi-<5mm (multiple)', 'pi-5-10mm', 'pi->10mm', 
                    'pb-none', 'pb-<5mm (single)', 'pb-<5mm (multiple)', 'pb-5-10mm', 'pb->10mm',
                    'cn-none', 'cn-<5mm (single)', 'cn-<5mm (multiple)', 'cn-5-10mm', 'cn->10mm',
                    ]

    label_features = ['alines', 'blines', 'pleural_break', 'pleural_indent', 'consolidation', ]

    lung_severity_scores = ['score-0', 'score-1', 'score-2', 'score-3']


    with open(dataset_split_dict_path, 'r') as json_file:
        dataset_split_dict = json.load(json_file)
            

    # # label_dict_path = 'user_label_ggare_0.json'
    # # label_dict_path = 'user_label_tsmNet_2.json'
    # # label_dict_path = 'user_label_tsmNet_3.json'
    # with open(label_dict_path, 'r') as json_file:
    #     test_label_dict = json.load(json_file)
    test_video_list, test_label_dict = loadPredLabels(test_y_prob_path, dataset_split_dict, dataset_type = 'test', trial = trial)

    # # label_dict_path = 'user_label_tsmNet_3.json'
    # with open('train_' + label_dict_path, 'r') as json_file:
    #     train_label_dict = json.load(json_file)
    # # train_label_dict = test_label_dict
    train_video_list, train_label_dict = loadPredLabels(train_y_prob_path, dataset_split_dict, dataset_type = 'train', trial = trial)
    
    with open(gt_label_dict_path, 'r') as json_file:
        gt_label_dict = json.load(json_file)

    
    
    # train_video_list = []
    # [train_video_list.extend(s) for s in dataset_split_dict['train'].values()]

    # test_video_list = []
    # [test_video_list.extend(s) for s in dataset_split_dict['test'].values()]

    # train_label = []
    # [[train_label.append(train_label_dict[video][f]) for f in label_features] for video in train_video_list]
    train_label_ft = np.array([[train_label_dict[video][f] for f in label_features] for video in train_video_list])
    assert np.array(train_label_ft).shape == (len(train_video_list), 5, 5)
    train_label = train_label_ft.reshape(train_label_ft.shape[0], -1)

    # test_label = []
    # [[test_label_dict[video][f] for f in label_features] for video in test_video_list]
    test_label_ft = np.array([[test_label_dict[video][f] for f in label_features] for video in test_video_list])
    assert np.array(test_label_ft).shape == (len(test_video_list), 5, 5)
    test_label = test_label_ft.reshape(test_label_ft.shape[0], -1)

    gt_train_scores = []
    [gt_train_scores.append(gt_label_dict[video]['lung-severity']) for video in train_video_list]
    assert np.array(gt_train_scores).shape == (len(train_video_list), 4)
    gt_train_scores = np.array(gt_train_scores).argmax(1)

    gt_test_scores = []
    [gt_test_scores.append(gt_label_dict[video]['lung-severity']) for video in test_video_list]
    assert np.array(gt_test_scores).shape == (len(test_video_list), 4)
    gt_test_scores = np.array(gt_test_scores).argmax(1)

    gt_train_label_ft = np.array([[gt_label_dict[video][f] for f in label_features] for video in train_video_list])
    assert np.array(gt_train_label_ft).shape == (len(train_video_list), 5, 5)
    gt_train_label = gt_train_label_ft.reshape(gt_train_label_ft.shape[0], -1)

    gt_test_label_ft = np.array([[gt_label_dict[video][f] for f in label_features] for video in test_video_list])
    assert np.array(gt_test_label_ft).shape == (len(test_video_list), 5, 5)
    gt_test_label = gt_test_label_ft.reshape(gt_test_label_ft.shape[0], -1)

    train_label_ft_acc = (train_label_ft == gt_train_label_ft).mean(axis = (0, 2))
    print(f"Train feature acc = {[(f,v) for f, v in zip(label_features, train_label_ft_acc)]}")

    test_label_ft_acc = (test_label_ft == gt_test_label_ft).mean(axis = (0, 2))
    print(f"Test feature acc = {[(f,v) for f, v in zip(label_features, test_label_ft_acc)]}")

    #Fit ML model
    clf = svm.SVC()
    clf = clf.fit(train_label, gt_train_scores)

    ml_predictions = clf.predict(test_label)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'SVM ML model accuracy = {accuracy}')


    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_label, gt_train_scores)

    ml_predictions = clf.predict(test_label)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'Decision-Tree ML model accuracy = {accuracy}')

    text_representation = tree.export_text(clf)
    print(text_representation)

    fig = plt.figure(figsize = (10,5), dpi=3000)
    # fig = plt.figure()
    # tree.plot_tree(clf) 
    tree.plot_tree(clf, 
                   feature_names=label_names,  
                   class_names=lung_severity_scores,
                   filled=True)
    plt.savefig("Decision Tree T5.png", bbox_inches='tight', dpi = 3000)
    plt.show()