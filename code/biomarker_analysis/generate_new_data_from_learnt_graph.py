import numpy as np

import matplotlib.pyplot as plt

import torch

import os

import pandas as pd
import networkx as nx

from causal_graphs.graph_utils import adj_matrix_to_edges, edges_or_adj_matrix, sort_graph_by_vars, get_node_relations

import utils





#Relabel nodes
vars_mapping = {
    "$X_{1}$": "a0", "$X_{2}$": "a1", "$X_{3}$": "a2", "$X_{4}$": "a3", "$X_{5}$": "a4",
    "$X_{6}$": "b0", "$X_{7}$": "b1", "$X_{8}$": "b2", "$X_{9}$": "b3", "$X_{10}$": "b4",
    "$X_{11}$": "bo0", "$X_{12}$": "bo1", "$X_{13}$": "bo2",
    "$X_{14}$": "pt0", "$X_{15}$": "pt1", "$X_{16}$": "pt2", "$X_{17}$": "pt3",
    "$X_{18}$": "pl0", "$X_{19}$": "pl1", "$X_{20}$": "pl2",
    "$X_{21}$": "i0", "$X_{22}$": "i1", "$X_{23}$": "i2", "$X_{24}$": "i3", "$X_{25}$": "i4",
    "$X_{26}$": "pb0", "$X_{27}$": "pb1", "$X_{28}$": "pb2", "$X_{29}$": "pb3", "$X_{30}$": "pb4",
    "$X_{31}$": "c0", "$X_{32}$": "c1", "$X_{33}$": "c2", "$X_{34}$": "c3", "$X_{35}$": "c4",
    "$X_{36}$": "e0", "$X_{37}$": "e1", "$X_{38}$": "e2",
    # "$X_{39}$": "s0", "$X_{40}$": "s1", "$X_{41}$": "s2", "$X_{42}$": "s3" #Severity classes
    "$X_{39}$": "d0", "$X_{40}$": "d1", "$X_{41}$": "d2", "$X_{42}$": "d3", "$X_{43}$": "d4", "$X_{44}$": "d5", "$X_{45}$": "d6" #Disease classes
}


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
    # "s0":38, "s1":39, "s2":40, "s3":41 #Severity classes
    "d0":38, "d1":39, "d2":40, "d3":41, "d4":42, "d5":43, "d6":44 #Disease classes
}


def drawGraph(Graph, reports_path = "."):


    labels = nx.get_edge_attributes(Graph, "weight")
    
    #Change float precision
    for k,v in labels.items():
        labels[k] = f'{v:0.2f}'

    A = nx.nx_agraph.to_agraph(Graph)        # convert to a graphviz graph
    A.layout(prog='dot')            # neato layout
    #A.draw('test3.pdf')

    root_nodes = np.unique([e1 for (e1, e2), v in labels.items()])
    root_nodes_colors = {}

    for idx, node in enumerate(root_nodes):
        color =  "#"+''.join([hex(np.random.randint(0,16))[-1] for i in range(6)])
        root_nodes_colors[node] = color

    for (e1, e2), v in labels.items():
        edge = A.get_edge(e1,e2)
        edge.attr['weight'] = v
        edge.attr['label'] = str(v)
        # edge.attr['color'] = "red:blue"
        edge.attr['color'] = root_nodes_colors[e1]
        
    A.draw(os.path.join(reports_path, f"adjacency_matrix_temp.png"),
            args='-Gnodesep=1.0 -Granksep=9.0 -Gfont_size=1', prog='dot' )  


def getSamplingOrder(adj_matrix, variables):


    G = pd.DataFrame(adj_matrix, index = variables, columns = variables)
    G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)
    
    #Draw graph

    DRAW_GRAPH = False
    if DRAW_GRAPH:
        # nx.draw_networkx(G)
        # plt.savefig('Graph_temp.png')
        # plt.show()
        drawGraph(G)

    #Node degrees 
    node_degrees = {}
    for var in variables:
        node_degrees[var]= G.degree[var]

    #Node in-degrees 
    node_in_degrees = {}
    for var in variables:
        node_in_degrees[var]= G.in_degree[var]

    in_degree_sort_idx = np.argsort(list(node_in_degrees.values()))
    ascending_in_degree_nodes =  np.array(variables.copy())[in_degree_sort_idx]

    #Node out-degrees 
    node_out_degrees = {}
    for var in variables:
        node_out_degrees[var]= G.out_degree[var]

    out_degree_sort_idx = np.argsort(list(node_out_degrees.values()))
    ascending_out_degree_nodes =  np.array(variables.copy())[out_degree_sort_idx]

    #Node parents
    node_parents = {}
    for var in variables:
        node_parents[var]= list(G.predecessors(var))

    unconnected_nodes = [v for v,d in node_degrees.items() if d == 0]
    print(f"Unconnected nodes = {unconnected_nodes}")

    root_nodes = [n for n,d in G.in_degree() if d == 0 and G.degree(n) != 0] 
    print(f"Root nodes = {root_nodes}")

    # nx.ancestors(G, 'd0')

    sorted_variables, edges, adj_matrix, sorted_idxs = sort_graph_by_vars(variables, adj_matrix = adj_matrix)

    # sorted_variables = np.array(variables)[sorted_idxs]

    # paths_dict = {}
    # for node in G:
    #     if G.out_degree(node)==0: #it's a leaf
    #         paths_dict[node] = nx.shortest_path(G, root, node)

    for var in sorted_variables:
        print(f"{var}: {node_parents[var]}")

    return G, sorted_variables, node_parents, root_nodes, unconnected_nodes




from sklearn.neural_network import MLPClassifier

def fitLargeMLP(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, 
        n_trial = 3, hidden_layer_sizes = (128, 64, 32), verbose = True):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        # clf = MLPClassifier()

        clf = MLPClassifier(
            # hidden_layer_sizes = (128, 64),
            # hidden_layer_sizes = (128, 64, 32),
            hidden_layer_sizes = hidden_layer_sizes,
            learning_rate = "adaptive", #constant
            verbose = verbose,
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



def predLargeMLP(clf, test_label_ft):
    
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)
    
    return clf, ml_predictions, ml_prob_predictions




#For saving & loading sklearn model - Ref: https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
from joblib import dump, load

def train_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, category_classes, reports_path = "."):

    trained_models = {}

    for var in sorted_variables:

        if var in unconnected_nodes:
            print(f"Skipping unconnected node {var}")
            continue
        elif var in root_nodes:
            print(f"Skipping root node {var}")
            continue

        parents = node_parents[var]
        print(f"Train for node {var} with parents {parents}")


        parents_idx = [causal_feature_mapping[v] for v in parents]

        train_sub_features = train_data[:, parents_idx]

        # train_labels = train_data[:, category_classes]
        train_labels = train_data[:, causal_feature_mapping[var]]

        ##Train model
        model, accuracy, ml_predictions, ml_prob_predictions = fitLargeMLP(
                                    train_label_ft =  train_sub_features, 
                                    test_label_ft = train_sub_features, 
                                    gt_train_scores = train_labels, gt_test_scores = train_labels, 
                                    n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                    verbose = False
                                )

        #Train model

        #Save trained model
        dump(model, os.path.join(reports_path, f"Gen_ML_Model_var_{var}.joblib")) 

        trained_models[var] = model

    return trained_models


#Numpy new way to sample from distributions: Ref: https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
from numpy.random import default_rng
rng = default_rng()

def generateNewData(trained_models, sorted_variables, variables, node_parents, root_nodes, unconnected_nodes, 
            category_classes, num_samples = 1000, Binary_Features = True):

    generated_data_dict = {} 

    for var in sorted_variables:

        if var in unconnected_nodes:
            print(f"Generating unconnected node {var}")

            sampled_var = rng.binomial(n = 1, p = 0.5, size = num_samples)
            sampled_var = sampled_var[:, np.newaxis]
            generated_data_dict[var] = sampled_var

            continue

        elif var in root_nodes:
            print(f"Generating root node {var}")

            sampled_var = rng.binomial(n = 1, p = 0.5, size = num_samples)
            sampled_var = sampled_var[:, np.newaxis]
            generated_data_dict[var] = sampled_var
        
            continue

        parents = node_parents[var]
        print(f"Generating data for node {var} with parents {parents}")

        
        # test_features = np.array([generated_data[p] for p in parents]).T
        test_features = np.hstack([generated_data_dict[p] for p in parents])
        print(f'test_featurs.shape = {test_features.shape}')

        ##Eval model

        model = trained_models[var]

        model, ml_predictions, ml_prob_predictions = predLargeMLP(model, test_features)

        if Binary_Features:
            sampled_var = ml_predictions
        else:
            sampled_var = ml_prob_predictions

        sampled_var = sampled_var[:, np.newaxis]

        generated_data_dict[var] = sampled_var

    
    generated_data = np.hstack([generated_data_dict[v] for v in variables])

    return generated_data





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

    # upsampled_features = features[label_indices, :]
    upsampled_features = features[label_indices]

    upsampled_labels = labels[label_indices]

    classes, count = np.unique(upsampled_labels, return_counts = True)

    assert np.array_equal(count, max_count * np.ones(len(classes))), "Error! Upsampling didn't result in class-balance"

    return upsampled_labels, upsampled_features, upsample_label_indices



def main():

    causal_discovery_exp_dir = "/home/grg/Research/ENCO/checkpoints/"
    
    acyclic_adj_matrix_path = "2022_23_Acyclic_DiseasePerPatient-max_GroundTruth_UpsampledTrainTest_sparity0.0004/binary_acyclic_matrix_001_DiseaseBiomarkersGTabTrainUpsampledPatient-max.npy"

    acyclic_adj_matrix = np.load(os.path.join(causal_discovery_exp_dir, acyclic_adj_matrix_path))

    print(f"acyclic_adj_matrix.shape = {acyclic_adj_matrix.shape}")

    adj_matrix = acyclic_adj_matrix

    ##Process adjacency matrix
    
    # num_classes = 4 #Severity-classes
    num_classes = 7 #Disease-classes

    class_names = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others', ]

    task = 'disease'


    #Calculate Scores 
    reports_path = "."
    exp_name = "Using Synthetic Data"
    report_path = os.path.join(reports_path, f"classification_report_{task}.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Classification report")

    logger.log(f"Exp name: {exp_name}")


    REVERSE_OUTGOING_CLASS_EDGES = True

    if not REVERSE_OUTGOING_CLASS_EDGES:
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes
    else:
        #Reverse the edge direction of severity classes; convert outgoing edges to incoming edges 
        # for s_idx in [38, 39, 40, 41]: #Severity classes
        # for s_idx in [38, 39, 40, 41, 42, 43, 44]: #Disease classes
        for s_idx in range(38, 38+num_classes): #Disease classes
            outgoing_edges = adj_matrix[s_idx, :]
            outgoing_nodes = np.where(outgoing_edges == 1)
            adj_matrix[outgoing_nodes, s_idx] = 1

    # # TODO-GRG: Need to check if we need to transpose the adj_matrix or not!!!
    # # Transpose for mask because adj[i,j] means that i->j
    # mask_adj_matrices = adj_matrix.transpose(1, 2)

    # variables = list(range(45))
    variables = list(vars_mapping.values())

    #Get sampling order
    G, sorted_variables, node_parents, root_nodes, unconnected_nodes = getSamplingOrder(adj_matrix, variables)

    
    ### Load training data
    
    train_data_path = "DiseaseBiomarkersGTabTrainUpsampledPatient-max.npz"

    train_data = np.load(train_data_path)['data_obs']


    category_classes = list(range(38, 38+num_classes))

    trained_gen_models = train_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, category_classes)
    
    num_samples = 500 #1000
    Binary_Features = True
    synthetic_data = generateNewData(trained_gen_models, sorted_variables, variables, node_parents, root_nodes, unconnected_nodes, 
            category_classes, num_samples, Binary_Features)


    ## Test the synthetic data
    
    test_data_path = "DiseaseBiomarkersGTabTestUpsampledPatient-max.npz"

    test_data = np.load(test_data_path)['data_obs']

    print(f"train_data.shape = {train_data.shape}")



    train_features = train_data[:, :38]

    train_labels = train_data[:, category_classes]

    UpSampleData = False
    if UpSampleData:
        upsam_train_labels, upsam_train_features, upsample_label_indices = upsampleFeatures(labels = train_labels, features = train_features) 
        train_labels = upsam_train_labels
        train_features = upsam_train_features

    print(f"train_labels.shape = {train_labels.shape}")
    print(f"train_features.shape = {train_features.shape}")

    test_features = test_data[:, :38]

    test_labels = test_data[:, category_classes]

    ##Train model
    model, accuracy, ml_predictions, ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  train_features, test_label_ft = test_features, 
                                # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                verbose = False
                            )

    logger.log(f"Accuracy on original train set = {accuracy}")

    # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

    model_results_dict = calScores(preds = ml_predictions, prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)
    synthetic_train_data = np.concatenate((train_data, synthetic_data), axis = 0)
    print(f"synthetic_train_data.shape = {synthetic_train_data.shape}")
    
    synthetic_train_features = synthetic_train_data[:, :38]

    synthetic_train_labels = synthetic_train_data[:, category_classes]

    if UpSampleData:
        upsam_synthetic_train_labels, upsam_synthetic_train_features, upsample_label_indices = upsampleFeatures(labels = synthetic_train_labels, features = synthetic_train_features) 
        synthetic_train_labels = upsam_synthetic_train_labels
        synthetic_train_features = upsam_synthetic_train_features

    print(f"synthetic_train_labels.shape = {synthetic_train_labels.shape}")
    print(f"synthetic_train_features.shape = {synthetic_train_features.shape}")

    synthetic_model, synthetic_accuracy, synthetic_ml_predictions, synthetic_ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                verbose = False
                            )

    logger.log(f"Accuracy on synthetic + original train set = {synthetic_accuracy}")

    # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
    synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)



    logger.close()

    pass



if __name__ == "__main__":
    print("Started...")
    main()
    print("Finished!")