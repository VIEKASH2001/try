import numpy as np

from matplotlib import pyplot as plt

import json

import os

import sys
sys.path.append("./code")
import utils


from sklearn import tree

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

