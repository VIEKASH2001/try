import numpy as np

from matplotlib import pyplot as plt

import json

import os

from sklearn import tree, svm, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

#For saving & loading sklearn model - Ref: https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
from joblib import dump, load

from feature_pred_analyser import generateModelFeaturePreds

import sys
sys.path.append("./code")
import utils



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

    report_file.write(f"ML model accuracy = {model_accuracy} \n")

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



def getFeatures(dataset_split_dict, train_folds, test_folds, train_label_dict, test_label_dict, \
    label_features, task, SF_category_dict, Disease_category_dict, BiomarkerModel, ):



    train_video_list = []
    # [train_video_list.extend(s) for s in dataset_split_dict['train'].values()]
    # [[train_video_list.extend(v) for k, v in fold_data.items() if k in train_folds] for fold_data in dataset_split_dict["train"].values()]
    [train_video_list.extend(dataset_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in train_folds]
    
        
    test_video_list = []
    # [test_video_list.extend(s) for s in dataset_split_dict['test'].values()]
    [test_video_list.extend(dataset_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in test_folds]

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



def decisonTreePostAnalysis(clf, ml_predictions, accuracy, train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list, \
    exp_name, task, reports_path, label_names, lung_severity_scores, test_label_dict, label_dict_path, plotDecisionTree = False, BiomarkerModel = True):


    text_representation = tree.export_text(clf)
    print(text_representation)

    if BiomarkerModel:
        showDecisionPath(clf, X_test = test_label_ft, feature_names = label_names,  
                    class_names = lung_severity_scores, 
                    predictions = ml_predictions, targets = gt_test_scores, 
                    sample_id = 0, model_accuracy = accuracy, 
                    report_path = os.path.join(reports_path, f"DecisionTree_report_for_{exp_name}_{task}.txt"))


        if task == "lung-severity":
            updated_test_label_dict = updateDecisionPathToJson(clf, X_test = test_label_ft, feature_names = label_names,  
                        class_names = lung_severity_scores, 
                        predictions = ml_predictions, targets = gt_test_scores, 
                        test_video_list = test_video_list, test_label_dict = test_label_dict)
            
            #Save updated_test_label

            updated_test_label_dict_path = label_dict_path.replace("_test.json", "_test_updated.json")

            updated_test_label_json = json.dumps(updated_test_label_dict, indent=4)
            f = open(updated_test_label_dict_path, "w")
            f.write(updated_test_label_json)
            f.close()

        if plotDecisionTree:

            fig = plt.figure(figsize = (10,5), dpi=3000)
            # fig = plt.figure()
            # tree.plot_tree(clf) 
            tree.plot_tree(clf, 
                        feature_names=label_names,  
                        class_names=lung_severity_scores,
                        filled=True)
            plt.savefig(os.path.join(reports_path, f"Decision Tree {label_dict_path.split('/')[1]}_{task}.png"), bbox_inches='tight', dpi = 3000)
            plt.show()
            # plt.clf()
            # plt.close()


def fitDecisionTree(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (DecisionTree) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (DecisionTree) accuracy = {accuracy}')


    
    return clf, accuracy, ml_predictions, ml_prob_predictions


def fitSVM(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        # clf = svm.SVC()
        clf = svm.SVC(probability = True) #Enable probability predictions
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (SVM) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (SVM) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions



def fitRandomForest(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = RandomForestClassifier(n_estimators = 100) #The number of trees in the forest (default 100).
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (RandomForest) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (RandomForest) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions


def fitAdaBoost(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        # clf = AdaBoostClassifier(n_estimators = 100) #The maximum number of estimators at which boosting is terminated (default 50).
        clf = AdaBoostClassifier(n_estimators = 50) #The maximum number of estimators at which boosting is terminated (default 50).
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (AdaBoost) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (AdaBoost) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions


def fitNearestNeighbours(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        # clf = neighbors.KNeighborsClassifier(n_neighbors = 10) #Number of neighbors to use by default for kneighbors queries (default 5).
        clf = neighbors.KNeighborsClassifier(n_neighbors = 5) #Number of neighbors to use by default for kneighbors queries (default 5).
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (NearestNeighbours) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (NearestNeighbours) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions


def fitMLP(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = MLPClassifier()
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (MLP) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (MLP) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions


def fitLargeMLP(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = MLPClassifier(
            # hidden_layer_sizes = (128, 64),
            hidden_layer_sizes = (128, 64, 32),
            learning_rate = "adaptive", #constant
            verbose = True,
        )
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (MLP Large) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (MLP Large) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions



from dso import DeepSymbolicRegressor

def fitDSR(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        config = {
                "task" : {
                    "task_type" : "regression",
                    # "function_set" : ["const", 0.5, "add", "sub", "mul", "div", "neg", "sin", "cos", "exp", "log"],
                    # "function_set" : ["const", 0.3, 0.5, 0.8, "add", "sub", "mul", "div", "neg",],
                    # "function_set" : ["add", "sub", "mul", "div", "neg", "max", "min", "abs"],
                    # "function_set" : ["add", "sub", "mul", "div", "neg", "max", "min",],
                    # "function_set" : ["add", "sub", "mul", "div", "neg", ],
                    # "function_set" : ["add", "sub", "mul", "div", "neg", 0.3, 0.5, 0.8, 0, 1, 1.2, -0.2],
                    "function_set" : ["add", "sub", "mul", "div", "neg", "abs", 0.3, 0.5, 0.8, 0, 1, 1.2, -0.2],
                    # "function_set" : ["const", 0.3, 0.5, 0.8, "add", "sub", "mul", "div", "neg", "max", "min", "abs"],
                    "decision_tree_threshold_set" : [-0.05, 0.0, 0.01, 0.3, 0.5, 0.8, 0, 1, 1.2, -0.2],
                },
                "training" : {
                        # "n_samples" : 2000000,
                        # "batch_size" : 500,
                        # "epsilon" : 0.02,
                        # // Recommended to set this to as many cores as you can use! Especially if
                        # // using the "const" token.
                        "n_cores_batch" : 20, #1, (default = 1)
                    },
                "prior": {
                        # // Memory sanity value. Limit strings to size 256
                        # // This can be set very high, but it runs slower.
                        # // Max value is 1000. 
                        "length" : {
                            "min_" : 4,
                            "max_" : 256,
                            "on" : True
                        },
                        # // Memory sanity value. Have at most 10 optimizable constants. 
                        # // This can be set very high, but it runs rather slow. 
                        "repeat" : {
                            "tokens" : "const",
                            "min_" : None,
                            "max_" : 10,
                            "on" : True
                        },
                        "inverse" : {
                            "on" : True
                        },
                        "trig" : {
                            "on" : True
                        },
                        "const" : {
                            "on" : True
                        },
                        "no_inputs" : {
                            "on" : True
                        },
                        "uniform_arity" : {
                            "on" : False
                        },
                        "soft_length" : {
                            "loc" : 10,
                            "scale" : 5,
                            "on" : True
                        }
                    }
            }
        clf = DeepSymbolicRegressor(config = config)
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions.round() == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (MLP) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = ml_predictions
    ml_predictions = ml_predictions.round()
    # ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy =  (ml_predictions.round() == gt_test_scores).mean()
    print(f'ML model (MLP) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions


from dso import DeepSymbolicOptimizer

def fitDSO(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        config = {
                "task" : {
                    "task_type" : "regression",
                    # "function_set" : ["const", 0.5, "add", "sub", "mul", "div", "neg", "sin", "cos", "exp", "log"],
                    # "function_set" : ["const", 0.3, 0.5, 0.8, "add", "sub", "mul", "div", "neg",],
                    # "function_set" : ["const", 0.3, 0.5, 0.8, "add", "sub", "mul", "div", "neg", "max", "min", "abs"],
                    "function_set" : ["add", "sub", "mul", "div", "neg", "abs", 0.3, 0.5, 0.8, 0, 1, 1.2, -0.2],
                    "decision_tree_threshold_set" : [-0.05, 0.0, 0.01, 0.3, 0.5, 0.8, 0, 1, 1.2, -0.2],
                },
            }
        clf = DeepSymbolicOptimizer(config = config)
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions.round() == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (MLP) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = ml_predictions
    ml_predictions = ml_predictions.round()
    # ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy =  (ml_predictions.round() == gt_test_scores).mean()
    print(f'ML model (MLP) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions


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
                    exp_name, prefix, task, reports_path, label_features, label_names, lung_severity_scores, test_label_dict, label_dict_path, BiomarkerModel, onlyEnd2End):

    ##Fit ML model
    if onlyEnd2End:
        ml_models = {}
        ml_models["EndToEnd"] = end2endModelScores
    else:
        ml_models = {
                        "Decision Tree": fitDecisionTree, 
                        "SVM": fitSVM,
                        "Random Forest": fitRandomForest,
                        "AdaBoost": fitAdaBoost, 
                        "Nearest Neighbours": fitNearestNeighbours, 
                        "MLP": fitMLP,
                        "MLP Large": fitLargeMLP,
                        # "DSR": fitDSR,
                        # "DSO": fitDSO,
                    }
    
        #Added EndToEnd classifier for end-to-end model
        if BiomarkerModel or (not BiomarkerModel and prefix == ""):
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
                model_results_dict, accuracy = biomarkerModelScores(exp_name, exp_path, lung_severity_scores, test_label_ft, gt_label_dict, test_video_list, label_features, label_names, task, logger)
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


def biomarkerModelScores(exp_name, exp_path, class_names, test_label_ft, gt_label_dict, test_video_list, label_features, label_names, task, logger):


    gt_test_label_ft = np.array([np.hstack([gt_label_dict[video][f] for f in label_features]) for video in test_video_list])
    assert gt_test_label_ft.shape == (len(test_video_list), 38)

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

def fitTaskMLmodels(task, dataset_split_dict, train_folds, test_folds, train_label_dict, test_label_dict, label_features, 
        exp_name, exp_path, prefix, reports_path, label_names, label_dict_path,
        SF_category_dict, Disease_category_dict, upsampleTrainFeatures, BiomarkerModel, onlyEnd2End):


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
        label_features, task, SF_category_dict, Disease_category_dict, BiomarkerModel, )

        #Upsample train set for class balancing
        if upsampleTrainFeatures:
            gt_train_scores, train_label_ft, upsample_label_indices = upsampleFeatures(labels = gt_train_scores, features = train_label_ft) 
    
    results_dict = fitMLmodels(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list, train_folds, test_folds,
                    exp_name, prefix, task, reports_path, label_features, label_names, lung_severity_scores, test_label_dict, label_dict_path, BiomarkerModel, onlyEnd2End)
   
    return results_dict


def fitFoldMLmodels(train_folds, test_folds, exp_path, exp_name, prefix, dataset_split_dict_path, dataset_split_dict,
            label_features, label_names, SF_category_dict, Disease_category_dict, gt_label_dict,
            upsampleTrainFeatures, multi_task, use_probs, BiomarkerModel, onlyEnd2End):

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
    
    results_dict = {}

    ### Task-specific Code ###
    for task in tasks_list: 

        task_results_dict = fitTaskMLmodels(task, dataset_split_dict, train_folds, test_folds, train_label_dict, test_label_dict, label_features, 
            exp_name, exp_path, prefix, reports_path, label_names, label_dict_path,
            SF_category_dict, Disease_category_dict, upsampleTrainFeatures, BiomarkerModel, onlyEnd2End)

        results_dict[task] = task_results_dict
    
    utils.writeJson(results_dict, os.path.join(reports_path, f"classification_results_all_tasks.json"))

    return results_dict


def combineFoldResults(results_dict, folds_list, exp_name, exp_path, tasks_list, prefix, onlyEnd2End, float_precision = 2, percent_factor = 100):

    if onlyEnd2End:
        # ml_models = []
        ml_models = ["EndToEnd"]
    else:
        ml_models = ["Decision Tree", "SVM", "Random Forest", "AdaBoost", "Nearest Neighbours", "MLP", "MLP Large"]

        #Added EndToEnd classifier for end-to-end model
        if BiomarkerModel or (not BiomarkerModel and prefix == ""):
            ml_models = ["EndToEnd"] + ml_models


    #Combine results 
    report_path = os.path.join(exp_path, f"{exp_name}{prefix}_combined_exp_report.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Combined Exp Report")

    logger.log(f"Exp name: {exp_name}")

    
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

             

    datadir = "/data1/datasets/LSU-Large-Dataset/"

    dataset_split_dict_path = os.path.join(datadir, "dataset_split_equi_class_R1.json")
    dataset_split_dict = utils.readJson(dataset_split_dict_path)

    SF_category_dict = utils.readJson(path = os.path.join(datadir, "SF_category_dict.json"))
    Disease_category_dict = utils.readJson(path = os.path.join(datadir, "Disease_category_dict.json"))

    gt_label_dict = utils.readJson(os.path.join(datadir, "user_label_ggare_2.json"))
 
    exp_path = "/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps"
    

    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_PlMl_T1_C"     
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1_C"       
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1a_C"     
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_C"
    # exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_RN34_T1_C"
      
    # exp_name = "GroundTruth_BiomarkerFeatures"

    exp_name = "tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2"
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

    # exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2"
    # exp_name = "tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2copy"
    # exp_name = "tsm_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_T2"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T1"
    # exp_name = "tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T2"

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

    # prefix = ""
    prefix = "_dso"
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
    
    onlyEnd2End = False
    
    BiomarkerModel = True #True

    MULTI_TASK = False
    useProbs = True
    upsampleTrainFeatures = True #True

    useGTFeatures_forTrain = False
    useGTFeatures_forTest = False

    # tasks_list = ["lung-severity", "sf-ratio", "diagnosis"]
    tasks_list = ["lung-severity"]
    # tasks_list = ["sf-ratio"]
    # tasks_list = ["diagnosis"]


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

        fold_results_dict = fitFoldMLmodels(train_folds, test_folds, exp_path, fold_exp_name, prefix, dataset_split_dict_path, dataset_split_dict,
            label_features, label_names, SF_category_dict, Disease_category_dict, gt_label_dict,
            upsampleTrainFeatures, multi_task = MULTI_TASK, use_probs = useProbs, BiomarkerModel = BiomarkerModel, onlyEnd2End = onlyEnd2End)


        results_dict[val_fold] = fold_results_dict
    
    utils.writeJson(results_dict, os.path.join(exp_path, f"{exp_name}{prefix}_classification_results.json"))

    combineFoldResults(results_dict, folds_list, exp_name, exp_path, tasks_list, prefix, onlyEnd2End)

    print("finished!")
