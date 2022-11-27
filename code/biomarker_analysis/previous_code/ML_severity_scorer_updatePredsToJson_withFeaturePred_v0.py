import numpy as np

from matplotlib import pyplot as plt

import json

import os

from sklearn import tree

from feature_pred_analyser import generateModelFeaturePreds


def showTreeStructure(clf, feature_names, class_names):

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("\n\n")

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with pred {pred}.".format(
                    space=node_depth[i] * "\t", node=i, pred = class_names[value[i].argmax()]
                )
            )
        else:
            print(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] {feature_nm} <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    feature_nm = feature_names[feature[i]],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )
    
    
    print("\n\n")

    return feature, threshold, value


def showDecisionPath(clf, X_test, feature_names, class_names, predictions, targets, sample_id = 0, model_accuracy = -1, report_path = "Decison_Tree_report.txt"):

    feature, threshold, value = showTreeStructure(clf, feature_names, class_names)

    # getDecisionPath_v0(clf, X_test, feature, threshold, value, feature_names, class_names, predictions, targets, sample_id)

    # getDecisionPath(clf, X_test, feature, threshold, value, feature_names, class_names, predictions, targets, sample_id)


    report_file = open(report_path, 'w')
    report_file.write(f'\n')
    report_file.write(f"{report_path.split('/')[-1].split('.')[0]} \n")

    report_file.write(f"ML model accuracy = {accuracy} \n")

    for id in range(len(predictions)):

        getDecisionPath(clf, X_test, feature, threshold, value, feature_names, class_names, predictions, targets, sample_id = id, report_file = report_file)

    report_file.close()


def updateDecisionPathToJson(clf, X_test, feature_names, class_names, predictions, targets, test_video_list, test_label_dict):

    feature, threshold, value = showTreeStructure(clf, feature_names, class_names)

    for id, video in enumerate(test_video_list):

        ml_pred, decsionPath = getDecisionPathTxt(clf, X_test, feature, threshold, value, feature_names, class_names, predictions, targets, sample_id = id)

        lung_severity = [0, 0, 0, 0]
        lung_severity[predictions[id]] = 1

        test_label_dict[video]['lung-severity'] = lung_severity
        test_label_dict[video]['unusual_findings'] = decsionPath

    return test_label_dict

def getDecisionPathTxt(clf, X_test, feature, threshold, value, feature_names, class_names, predictions, targets, sample_id = 0):

    node_indicator = clf.decision_path(X_test)
    leaf_id = clf.apply(X_test)

    # sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    ml_pred = class_names[predictions[sample_id]]
    ml_target = class_names[targets[sample_id]]

    decision_path_str = f"Target {ml_target}; Pred {ml_pred}; Rules =\n"
    
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            
            decision_path_str += f" therefore {ml_pred}"
            continue

        # check if value of the split feature for sample 0 is below threshold
        if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "No"
        else:
            threshold_sign = "Yes"

        feature_nm = feature_names[feature[node_id]]

        decision_path_str += f" {threshold_sign} {feature_nm};\n"

    print(f"\n{decision_path_str}\n")
    
    return ml_pred, decision_path_str


def getDecisionPath_v0(clf, X_test, feature, threshold, value, feature_names, class_names, predictions, targets, sample_id = 0):

    node_indicator = clf.decision_path(X_test)
    leaf_id = clf.apply(X_test)

    # sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    print("\n\n")

    print("Rules used to predict test sample {id} with pred {ml_pred} & target {ml_target}:\n".format(
            id=sample_id, 
            ml_pred = class_names[predictions[sample_id]], 
            ml_target = class_names[targets[sample_id]]
        )
    )
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            print(
                "leaf node {node} : pred {pred}".format(
                    node=node_id,
                    pred=class_names[value[node_id].argmax()]
                )
            )
            continue

        # check if value of the split feature for sample 0 is below threshold
        if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print(
            "decision node {node} : (X_test[{sample}, {feature}] {feature_nm} = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                sample=sample_id,
                feature=feature[node_id],
                feature_nm = feature_names[feature[node_id]],
                value=X_test[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
            )
        )


    print("\n\n")


def getDecisionPath(clf, X_test, feature, threshold, value, feature_names, class_names, predictions, targets, sample_id = 0, report_file = None):

    node_indicator = clf.decision_path(X_test)
    leaf_id = clf.apply(X_test)

    # sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    print_str = "\n\nRules used to predict test sample {id} with pred {ml_pred} & target {ml_target}:\n".format(
            id=sample_id, 
            ml_pred = class_names[predictions[sample_id]], 
            ml_target = class_names[targets[sample_id]]
        )
    
    print(print_str)
    if report_file is not None:
        report_file.write(f"{print_str}\n")
    
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            
            print_str = "leaf node {node} : pred {pred}".format(
                    node=node_id,
                    pred=class_names[value[node_id].argmax()]
                )
            
            print(print_str)
            if report_file is not None:
                report_file.write(f"{print_str}\n")

            continue

        # check if value of the split feature for sample 0 is below threshold
        if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "No"
        else:
            threshold_sign = "Yes"

        print_str = "decision node {node} : {inequality} {feature_nm} [value = {value}] ".format(
                node=node_id,
                feature_nm = feature_names[feature[node_id]],
                value=X_test[sample_id, feature[node_id]],
                inequality=threshold_sign,
            )

        print(print_str)
        if report_file is not None:
            report_file.write(f"{print_str}\n")


if __name__ == '__main__':

   
    train_folds = ["B", "C"]
    # train_folds = ["A", "C"]

    reports_path = "reports"

    # dataset_split_dict_path = "/data1/datasets/LSU-Dataset/dataset_split_equi_class_R1.json"
    dataset_split_dict_path = "/data1/datasets/LSU-Dataset/dataset_split_equi_class_R3.json"
    with open(dataset_split_dict_path, 'r') as json_file:
        dataset_split_dict = json.load(json_file)

 
    # exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps"
    exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps_R3"
    
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
    # exp_name = "tsm_LSU-Dataset_MultiTask_AP_EquiTS_PlMl_T2_A"            
    # exp_name = "tsm_LSU-Dataset_MultiTask_AP_EquiTS_PlMl_T3_A"              
    # exp_name = "tsm_LSU-Dataset_BioM_AP_EquiTS_PlMl_T2_A"          
    # exp_name = "tsm_LSU-Dataset_BioM_AP_EquiTS_PlMl_T1_A"     
    exp_name = "tsm_LSU-Dataset_BioM_AP_EquiTS_PlMl_T2_A"
    # exp_name = "tsm_LSU-Dataset_MultiTask_AP_EquiTS_PlMl_T1_A"         

    MULTI_TASK = False
    useProbs = True

    useGTFeatures_forTrain = False
    useGTFeatures_forTest = False

    # label_dict_path = "/data1/datasets/LSU-Dataset/user_label_gautamgare_12.json"
    # label_dict_path = 'user_label_tsmNet_1.json'
    # label_dict_path = 'user_label_tsmNet_2a.json'
    # label_dict_path = 'user_label_tsmNet_4b.json'
    # label_dict_path = 'user_label_tsmNet_4a.json'
    # label_dict_path = 'user_label_tsmNet_5a.json'
    # label_dict_path = 'user_label_tsmNet_6a.json'
    # label_dict_path = 'user_label_tsmNet_7a.json'
    # label_dict_path = 'user_label_tsmNet_8a.json'
    # label_dict_path = 'user_label_tsmNet_9a.json'
    # label_dict_path = 'user_label_tsmNet_10a.json'
    label_dict_path = f'user_label_{exp_name}_test.json'

    label_dict_path = os.path.join(reports_path, label_dict_path)

    generateModelFeaturePreds(dataset_split_dict_path, exp_path, exp_name, label_dict_path, train_folds, multi_task = MULTI_TASK, use_probs = useProbs)


    with open(label_dict_path, 'r') as json_file:
        test_label_dict = json.load(json_file)


    # label_dict_path = "/data1/datasets/LSU-Dataset/user_label_gautamgare_12.json"
    # label_dict_path = 'user_label_tsmNet_1.json'
    train_label_dict_path = label_dict_path.replace("_test.json", "_train.json")
    with open(train_label_dict_path, 'r') as json_file:
        train_label_dict = json.load(json_file)
    # train_label_dict = test_label_dict
    
    gt_label_dict_path = "/data1/datasets/LSU-Dataset/user_label_gautamgare_12.json"
    with open(gt_label_dict_path, 'r') as json_file:
        gt_label_dict = json.load(json_file)

    #Correct video names
    corrected_gt_label_dict = {}
    for v, l in gt_label_dict.items():
        corrected_gt_label_dict[v.split('/')[-1]] = l
    gt_label_dict = corrected_gt_label_dict

    if useGTFeatures_forTrain:
        train_label_dict = gt_label_dict
    if useGTFeatures_forTest:
        test_label_dict = gt_label_dict
    
    label_names = [ 
                    'al-none', 'al-weak', 'al-bold', 'al-*also* stacked', 'al-*also* wide (> 2cm)',
                    'bl-none', 'bl-few (1-3)', 'bl-some (4-5)', 'bl-many|coalescing', "bl-\"white\" (no striations)",   
                    'pi-none', 'pi-<5mm (single)', 'pi-<5mm (multiple)', 'pi-5-10mm', 'pi->10mm', 
                    'pb-none', 'pb-<5mm (single)', 'pb-<5mm (multiple)', 'pb-5-10mm', 'pb->10mm',
                    'cn-none', 'cn-<5mm (single)', 'cn-<5mm (multiple)', 'cn-5-10mm', 'cn->10mm',
                    ]

    label_features = ['alines', 'blines', 'pleural_break', 'pleural_indent', 'consolidation', ]

    lung_severity_scores = ['score-0', 'score-1', 'score-2', 'score-3']

    train_video_list = []
    # [train_video_list.extend(s) for s in dataset_split_dict['train'].values()]
    [[train_video_list.extend(v) for k, v in fold_data.items() if k in train_folds] for fold_data in dataset_split_dict["train"].values()]

    test_video_list = []
    [test_video_list.extend(s) for s in dataset_split_dict['test'].values()]

    train_video_list = [v.split('/')[-1] for v in train_video_list]
    test_video_list = [v.split('/')[-1] for v in test_video_list]

    # #Exclude some train videos as they are not present in the model - #TODO-GRG: Need to fix this
    # print([ v for v in train_video_list if v not in list(train_label_dict.keys())])
    # # train_video_list = [v for v in train_video_list if v not in ['114/114(06.21.21)L1-120-6(C19, AKI, NSTEMI).avi', '50/50(03.15.21)L1-471-4(NSTEMI, CKD3, obesity).avi', '112/112(06.21.21)R1-300-5(HFpEF, ESRD, GIB).avi', '148/148(08.16.21)L1-192-7(C19, cholecystitis).avi', '21/21(04.25.21)L1-139-7(C19, chronic lung disease).avi']]
    # train_video_list = [v for v in train_video_list if v not in ['96/96(05.13.21)L1-388-5(CHF, HTN emergency).avi', '36/36(03.08.21)L1-329-5(biventricular failure, ESRD, HTN).avi', '21/21(03.12.21)R1-163-6(C19, chronic lung disease).avi', '50/50(03.15.21)L1-471-4(NSTEMI, CKD3, obesity).avi', '112/112(06.21.21)L1-300-5(HFpEF, ESRD, GIB).avi', '148/148(08.16.21)L1-192-7(C19, cholecystitis).avi', '21/21(04.25.21)L1-139-7(C19, chronic lung disease).avi']]
    # train_video_list = [v for v in train_video_list if v not in ['152/152(8.20.21)L1.avi', '61/61(03.22.21)R1-467-4(NSTEMI, COPD, HFpEF).avi', '114/114(06.21.21)L1-120-6(C19, AKI, NSTEMI).avi', '89/89(04.28.21)L1-327-5(CHF, CKD4, AS).avi', '123/123(07.09.21)R1-323-5(hypercapnic resp fail).avi', '21/21(03.22.21)R1-165-7(C19, chronic lung disease).avi']]
    # train_video_list = [v for v in train_video_list if v not in ['152(8.20.21)L1.avi', '61(03.22.21)R1-467-4(NSTEMI, COPD, HFpEF).avi', '114(06.21.21)L1-120-6(C19, AKI, NSTEMI).avi', '89(04.28.21)L1-327-5(CHF, CKD4, AS).avi', '123(07.09.21)R1-323-5(hypercapnic resp fail).avi', '21(03.22.21)R1-165-7(C19, chronic lung disease).avi']]
    
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
    # assert np.array(gt_train_scores).shape == (len(train_video_list), 4)
    gt_train_scores = np.array(gt_train_scores).argmax(1)
    assert gt_train_scores.shape[0] == len(train_video_list)

    gt_test_scores = []
    [gt_test_scores.append(gt_label_dict[video]['lung-severity']) for video in test_video_list]
    # assert np.array(gt_test_scores).shape == (len(test_video_list), 4)
    gt_test_scores = np.array(gt_test_scores).argmax(1)
    assert gt_test_scores.shape[0] == len(test_video_list)

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
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_label, gt_train_scores)

    ml_predictions = clf.predict(test_label)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model accuracy = {accuracy}')

    text_representation = tree.export_text(clf)
    print(text_representation)


    showDecisionPath(clf, X_test = test_label, feature_names = label_names,  
                   class_names = lung_severity_scores, 
                   predictions = ml_predictions, targets = gt_test_scores, 
                   sample_id = 0, model_accuracy = accuracy, 
                   report_path = os.path.join(reports_path, f"DecisionTree_report_for_{exp_name}.txt"))



    updated_test_label_dict = updateDecisionPathToJson(clf, X_test = test_label, feature_names = label_names,  
                   class_names = lung_severity_scores, 
                   predictions = ml_predictions, targets = gt_test_scores, 
                   test_video_list = test_video_list, test_label_dict = test_label_dict)
    
    #Save updated_test_label

    updated_test_label_dict_path = label_dict_path.replace("_test.json", "_test_updated.json")

    updated_test_label_json = json.dumps(updated_test_label_dict, indent=4)
    f = open(updated_test_label_dict_path, "w")
    f.write(updated_test_label_json)
    f.close()

    fig = plt.figure(figsize = (10,5), dpi=3000)
    # fig = plt.figure()
    # tree.plot_tree(clf) 
    tree.plot_tree(clf, 
                   feature_names=label_names,  
                   class_names=lung_severity_scores,
                   filled=True)
    plt.savefig(os.path.join(reports_path, f"Decision Tree {label_dict_path.split('/')[1]}.png"), bbox_inches='tight', dpi = 3000)
    plt.show()
    # plt.clf()
    # plt.close()