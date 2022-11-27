import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import os
import torchmetrics

import tqdm


import sys
sys.path.append("./code")
import utils

from biomarker_analysis.fit_ML_classsifer_multiFoldExp import calScores
import biomarker_analysis.ML_classifiers as ml


def loadData(reports_path, task):

    #Note: The train and test features are upsampled to ensure class balancing  
    train_label_ft = np.load(os.path.join(reports_path, f"{task}_train_label_ft.npy"))
    test_label_ft = np.load(os.path.join(reports_path, f"{task}_test_label_ft.npy"))
    gt_train_scores = np.load(os.path.join(reports_path, f"{task}_gt_train_scores.npy"))
    gt_test_scores = np.load(os.path.join(reports_path, f"{task}_gt_test_scores.npy"))
    train_video_list = np.load(os.path.join(reports_path, f"{task}_train_video_list.npy"))
    test_video_list = np.load(os.path.join(reports_path, f"{task}_test_video_list.npy"))
    upsample_label_indices = np.load(os.path.join(reports_path, f"{task}_upsample_label_indices.npy"))

    #Up-sample
    train_video_list = train_video_list[upsample_label_indices]
    assert train_label_ft.shape[0] == len(train_video_list), "Error! Missing input videos."

    return train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list


class CustomDataset(Dataset):

    def __init__(self, filenames, inputs, targets):
        
        self.filenames = filenames
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        input = self.inputs[idx]
        target = self.targets[idx]

        input = torch.tensor(input)
        target = torch.tensor(target)
        
        input = input.float()

        return filename, input, target


def train(model, train_dataloader, optimizer, criterion,
            num_classes, SparsifyDuringTrain = True,
            device = "cuda:1", num_epochs = 100):

    model.to(device)


    # for epoch in range(1, num_epochs + 1):
    for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
        
        train_acc = torchmetrics.Accuracy(num_classes = num_classes, average = "weighted", multiclass = True)
        train_acc.to(device)
        
        model.train()

        # for idx, (filenames, inputs, targets) in enumerate(tqdm.tqdm(train_dataloader)):
        for idx, (filenames, inputs, targets) in enumerate(train_dataloader):

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            #Cal acc
            acc = train_acc(outputs, targets)
            # print(f"Accuracy on batch {idx}: {acc}")
        

        # # metric on all batches using custom accumulation
        # acc = train_acc.compute()
        # print(f"Accuracy on all data: {acc}")

        #Sparsify Model
        if SparsifyDuringTrain:
            if epoch == num_epochs:
                model = sparsifyMLPDuringTrain(model, reintialize = False)
            else:
                model = sparsifyMLPDuringTrain(model)

    # metric on all batches using custom accumulation
    acc = train_acc.compute()
    print(f"Train accuracy on all data: {acc}")

    return model, acc


def trainHidden(model, train_dataloader, optimizer, criterion,
            num_classes, SparsifyDuringTrain = True,
            device = "cuda:1", num_epochs = 100):

    model.to(device)


    # for epoch in range(1, num_epochs + 1):
    for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
        
        train_acc = torchmetrics.Accuracy(num_classes = num_classes, average = "weighted", multiclass = True)
        train_acc.to(device)
        
        model.train()

        # for idx, (filenames, inputs, targets) in enumerate(tqdm.tqdm(train_dataloader)):
        for idx, (filenames, inputs, targets) in enumerate(train_dataloader):

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            # #Cal acc
            # acc = train_acc(outputs, targets)
            # # print(f"Accuracy on batch {idx}: {acc}")
        

        # # metric on all batches using custom accumulation
        # acc = train_acc.compute()
        # print(f"Accuracy on all data: {acc}")

        #Sparsify Model
        if SparsifyDuringTrain:
            if epoch == num_epochs:
                model = sparsifyMLPDuringTrain(model, reintialize = False)
            else:
                model = sparsifyMLPDuringTrain(model)

    # # metric on all batches using custom accumulation
    # acc = train_acc.compute()
    # print(f"Train accuracy on all data: {acc}")

    # return model, acc
    return model


def test(model, test_dataloader, criterion, task,
            class_names, reports_path, exp_name, previous_model = False,
            return_features = False, return_prob_preds = False, device = "cuda:1"):


    model.eval()


    with torch.no_grad():

        
        targets_list = []
        prob_preds_list = []
        features_list = []
        for idx, (filenames, inputs, targets) in enumerate(tqdm.tqdm(test_dataloader)):

            inputs = inputs.to(device)
            targets = targets.to(device)

            if previous_model:
                outputs = model(inputs, previous = True)
            else:
                if return_features:
                    outputs, features = model(inputs, return_features = True)
                    # features_list.append(features.detach().cpu().numpy())
                    features_list.append(torch.cat((inputs, features), dim = 1).detach().cpu().numpy())
                else:
                    outputs = model(inputs)

            loss = criterion(outputs, targets)

            targets_list.append(targets.detach().cpu().numpy())
            prob_preds_list.append(outputs.detach().cpu().numpy())


    targets_list = np.hstack(targets_list)
    prob_preds_list = np.vstack(prob_preds_list)
    prob_preds_list = torch.softmax(torch.tensor(prob_preds_list), dim = 1).numpy()
    preds_list = np.argmax(prob_preds_list, axis = 1)

    #Calculate Scores 
    report_path = os.path.join(reports_path, f"classification_report_{task}.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Classification report")

    logger.log(f"Exp name: {exp_name}")

    model_results_dict = calScores(preds = preds_list, prob_preds = prob_preds_list, 
            targets = targets_list, class_names = class_names, task = task, logger = logger)

    utils.writeJson(model_results_dict, os.path.join(reports_path, f"classification_results_{task}.json"))

    logger.close()

    model.train()


    print(f"Accuracy = {model_results_dict['accuracy']}")

    if return_features:
        features_list = np.vstack(features_list)
        
        if return_prob_preds:
            return features_list, prob_preds_list, model_results_dict['accuracy']
        
        return features_list, model_results_dict['accuracy']

    if return_prob_preds:
        return prob_preds_list, model_results_dict['accuracy']

    return model_results_dict['accuracy']


def sparsifyMLP(model):


    for name, param in model.named_parameters():

        # if "weight" in name:
        if "weight" in name:
            
            if "classifier" in name:
                print(f"Skipping sparsification of - {name}")
                continue

            tmp_param = param.detach()
            tmp_abs_param = tmp_param.abs()
            m_param = tmp_abs_param.mean(dim=1)
            s_param = tmp_abs_param.std(dim=1)

            for t, tab, m, s in zip(tmp_param, tmp_abs_param, m_param, s_param):
                t[tab < m] = 0
                # t[tab < m + s] = 0

            # param.copy_(tmp_param)

    return model


def sparsifyMLPDuringTrain(model, reintialize = True):


    for name, param in model.named_parameters():

        # if "weight" in name:
        if "weight" in name:
            
            if "classifier" in name:
                print(f"Skipping sparsification of - {name}")
                continue

            tmp_param = param.detach()
            tmp_abs_param = tmp_param.abs()
            m_param = tmp_abs_param.mean(dim=1)
            s_param = tmp_abs_param.std(dim=1)

            init_param = tmp_param.clone()
            init_param = nn.init.xavier_uniform_(init_param)

            for t, i, tab, m, s in zip(tmp_param, init_param, tmp_abs_param, m_param, s_param):
                
                if reintialize:
                    # t[tab < m] = i[tab < m]
                    t[tab < m + s] = i[tab < m + s]
                else:
                    # t[tab < m] = 0
                    t[tab < m + s] = 0

            # param.copy_(tmp_param)

    return model




def main():

    exp_dir = "/home/grg/Research/DARPA-Pneumothorax/results/Diagnostic_Rules_Exps/tsm_LSU-Large-Dataset_Bio_Crop_upsampleVal_T1_C/reports_ml_tasks"
    # exp_dir = "/home/grg/Research/DARPA-Pneumothorax/results/Diagnostic_Rules_Exps/tsm_LSU-Large-Dataset_Bio_Crop_upsampleVal_T1_C/reports_ml_gt_tasks"

    # task = "lung-severity"
    task = "diagnosis"
 
    exp_name = "Trial-1"
    reports_path = os.path.join(exp_dir, exp_name)
    utils.createDirIfDoesntExists(reports_path)

    net_size = 100 #100 #50 #25

    Sparsify = False #This is necessarcy as after traain epoch sparsification we xavier initialize instead of zeroring 
    SparsifyDuringTrain = True

    FreezePreviousLayers = False

    #Load data
    train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list = loadData(reports_path = exp_dir, task = task)

    # class_names = ['score-0', 'score-1', 'score-2', 'score-3']
    class_names = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others']
    num_classes = len(class_names)
    input_features = 38

    assert train_label_ft.shape == (len(gt_train_scores), input_features), "Error! Feature len mismatch."
    assert test_label_ft.shape == (len(gt_test_scores), input_features), "Error! Feature len mismatch."

    train_dataset = CustomDataset(filenames = train_video_list, inputs = train_label_ft, targets = gt_train_scores)
    test_dataset = CustomDataset(filenames = test_video_list, inputs = test_label_ft, targets = gt_test_scores)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print_results = ""

    #Initialize model

    # model = nn.Sequential(
    #             nn.Linear(input_features, net_size), nn.ReLU(), #Input-layer
    #             # nn.Linear(100, 100), nn.ReLU(), #Hidden-layers
    #             nn.Linear(net_size, num_classes) #Output-layer
    #         ) 

    print_results += f"\n\n Initial model \n\n"

    model = mlpModel(input_features, num_classes, net_size)

    #Initialize optimizer

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    #Train the model
    model, train_acc = train(model, train_dataloader, optimizer, criterion, num_classes, SparsifyDuringTrain)

    print_results += f"\n Train acc = {train_acc} \n"

    #Test the model
    test_acc = test(model, test_dataloader, criterion, task,
            class_names, reports_path, exp_name)

    print_results += f"\n Test acc = {test_acc} \n"

    if Sparsify:

        model = sparsifyMLP(model)

        #Test the model
        test_acc = test(model, test_dataloader, criterion, task,
                class_names, reports_path, exp_name)
        
        print_results += f"\n Test acc after sparsification = {test_acc} \n"

    ## Decision Tree ##

    clf, accuracy, ml_predictions, ml_prob_predictions = ml.fitDecisionTree(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, n_trial = 3)


    train_features, train_acc = test(model, DataLoader(train_dataset, batch_size=64, shuffle=False), criterion, task,
            class_names, reports_path, exp_name, return_features = True)

    test_features, test_acc = test(model, test_dataloader, criterion, task,
            class_names, reports_path, exp_name, return_features = True)

    clf, accuracy, ml_predictions, ml_prob_predictions = ml.fitDecisionTree(train_features, test_features, gt_train_scores, gt_test_scores, n_trial = 3)


    ## Sequentially add layer and train ##
    new_model = seqModel(prev_model = model.cpu(), net_size = net_size)
    print_results += f"\n\n 1st injection \n\n"
    
    #Check if the weights are correctly copied
    assert np.all((new_model.new_classifier_layer.weight[:, :net_size].cpu() == model.classifier_layer.weight.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."
    assert np.all((new_model.new_classifier_layer.bias.cpu() == model.classifier_layer.bias.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."

    #Freeze previous model parameters
    if FreezePreviousLayers:
        for name, param in new_model.named_parameters():
            #Don't freeze the sequential classifier layer
            # if "new_" in name:
            if "new_" in name and 'prev_' not in name:
                print(f"Not freezing - {name}")
                continue

            param.requires_grad = False


    #Train the new model
    model = new_model


    def getModelPraramsToOptimize(model):
               
        params = filter(lambda p: p.requires_grad, model.parameters())

        return params

    optimizer = torch.optim.Adam(getModelPraramsToOptimize(model), lr = 1e-3)
    # optimizer = torch.optim.Adam(getModelPraramsToOptimize(model), lr = 5e-4)

    #Train the model
    model, train_acc = train(model, train_dataloader, optimizer, criterion, num_classes, SparsifyDuringTrain)

    print_results += f"\n Train acc = {train_acc} \n"

    #Test the model
    test_acc = test(model, test_dataloader, criterion, task,
            class_names, reports_path, exp_name)

    print_results += f"\n Test acc = {test_acc} \n"

    
    # #Test the previous model
    # test_acc = test(model, test_dataloader, criterion, task,
    #         class_names, reports_path, exp_name, previous_model = True)

    if Sparsify:
        
        model = sparsifyMLP(model)

        #Test the model
        test_acc = test(model, test_dataloader, criterion, task,
                class_names, reports_path, exp_name)

        print_results += f"\n Test acc after sparsification = {test_acc} \n"


    ## [2nd] - Sequentially add layer and train ##
    new_model = seq2ndModel(prev_model = model.cpu(), net_size = net_size)
    print_results += f"\n\n 2nd injection \n\n"
    
    #Check if the weights are correctly copied
    assert np.all((new_model.new_classifier_layer.weight[:, :2*net_size].cpu() == model.new_classifier_layer.weight.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."
    assert np.all((new_model.new_classifier_layer.bias.cpu() == model.new_classifier_layer.bias.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."

    #Freeze previous model parameters
    if FreezePreviousLayers:
        for name, param in new_model.named_parameters():
            #Don't freeze the sequential classifier layer
            if "new_" in name and 'prev_' not in name:
                print(f"Not freezing - {name}")
                continue

            param.requires_grad = False


    #Train the new model
    model = new_model


    def getModelPraramsToOptimize(model):
               
        params = filter(lambda p: p.requires_grad, model.parameters())

        return params

    optimizer = torch.optim.Adam(getModelPraramsToOptimize(model), lr = 1e-3)
    # optimizer = torch.optim.Adam(getModelPraramsToOptimize(model), lr = 5e-4)

    #Train the model
    model, train_acc = train(model, train_dataloader, optimizer, criterion, num_classes, SparsifyDuringTrain)

    print_results += f"\n Train acc = {train_acc} \n"

    #Test the model
    test_acc = test(model, test_dataloader, criterion, task,
            class_names, reports_path, exp_name)

    print_results += f"\n Test acc = {test_acc} \n"

    
    # #Test the previous model
    # test_acc = test(model, test_dataloader, criterion, task,
    #         class_names, reports_path, exp_name, previous_model = True)

    if Sparsify:
        
        model = sparsifyMLP(model)

        #Test the model
        test_acc = test(model, test_dataloader, criterion, task,
                class_names, reports_path, exp_name)

        print_results += f"\n Test acc after sparsification = {test_acc} \n"


    print(print_results)
    pass



class mlpModel(nn.Module):

    def __init__(self, input_features, num_classes, net_size):
        super(mlpModel, self).__init__()

        self.intial_layer = nn.Sequential(
                nn.Linear(input_features, net_size), nn.ReLU(), #Input-layer
            )
        # self.intial_layer = nn.Linear(input_features, net_size) #Input-layer
            

        self.classifier_layer = nn.Linear(net_size, num_classes) #Output-layer


    def forward(self, inputs, return_features = False):


        intial_out = self.intial_layer(inputs)

        if return_features:
            features = intial_out.clone()


        # intial_out = nn.functional.relu(intial_out)

        output = self.classifier_layer(intial_out)

        if return_features:
            return output, features

        return output


class seqModel(nn.Module):

    def __init__(self, prev_model, net_size):
        super(seqModel, self).__init__()

        self.prev_model = prev_model

        self.intial_layer = self.prev_model.intial_layer

        self.classifier_layer = self.prev_model.classifier_layer

        in_features = self.intial_layer[0].in_features
        previous_output = self.intial_layer[-2].out_features

        num_classes = self.classifier_layer.out_features

        self.new_layer = nn.Sequential(
                nn.Linear(previous_output + in_features, net_size), nn.ReLU(), #New-Hidden-layer
            )

        self.new_classifier_layer = nn.Linear(previous_output + net_size, num_classes)

        #Intialize with previous layer weights
        self.new_classifier_layer.weight[:, :net_size].data.copy_(self.classifier_layer.weight)
        self.new_classifier_layer.bias.data.copy_(self.classifier_layer.bias)

        #Check if the weights are correctly copied
        assert np.all((self.new_classifier_layer.weight[:, :net_size].cpu() == self.classifier_layer.weight.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."
        assert np.all((self.new_classifier_layer.bias.cpu() == self.classifier_layer.bias.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."

        pass

    def forward(self, inputs, previous = False):

        if previous:
            return self.prev_forward(inputs)

        intial_out = self.intial_layer(inputs)

        mid_out = self.new_layer(torch.cat((intial_out, inputs), dim = 1))

        output = self.new_classifier_layer(torch.cat((intial_out, mid_out), dim = 1))

        return output


    def prev_forward(self, inputs):

        intial_out = self.intial_layer(inputs)

        output = self.classifier_layer(intial_out)

        return output


class seq2ndModel(nn.Module):

    def __init__(self, prev_model, net_size):
        super(seq2ndModel, self).__init__()

        self.prev_model = prev_model

        self.intial_layer = self.prev_model.intial_layer

        self.second_layer = self.prev_model.new_layer

        self.classifier_layer = self.prev_model.new_classifier_layer

        in_features = self.intial_layer[0].in_features
        previous_output = self.intial_layer[-2].out_features

        second_output = self.second_layer[-2].out_features

        num_classes = self.classifier_layer.out_features

        self.new_layer = nn.Sequential(
                nn.Linear(second_output + previous_output + in_features, net_size), nn.ReLU(), #New-Hidden-layer
            )

        self.new_classifier_layer = nn.Linear(previous_output + second_output + net_size, num_classes)

        #Intialize with previous layer weights
        self.new_classifier_layer.weight[:, :2*net_size].data.copy_(self.classifier_layer.weight)
        self.new_classifier_layer.bias.data.copy_(self.classifier_layer.bias)

        #Check if the weights are correctly copied
        assert np.all((self.new_classifier_layer.weight[:, :2*net_size].cpu() == self.classifier_layer.weight.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."
        assert np.all((self.new_classifier_layer.bias.cpu() == self.classifier_layer.bias.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."

        pass

    def forward(self, inputs, previous = False):

        if previous:
            return self.prev_forward(inputs)

        intial_out = self.intial_layer(inputs)

        second_out = self.second_layer(torch.cat((intial_out, inputs), dim = 1))

        mid_out = self.new_layer(torch.cat((second_out, intial_out, inputs), dim = 1))

        output = self.new_classifier_layer(torch.cat((intial_out, second_out, mid_out), dim = 1))

        return output



    def prev_forward(self, inputs):

        intial_out = self.intial_layer(inputs)

        second_out = self.second_layer(torch.cat((intial_out, inputs), dim = 1))

        output = self.classifier_layer(torch.cat((intial_out, second_out), dim = 1))

        return output

    def prev_initial_forward(self, inputs):

        intial_out = self.intial_layer(inputs)

        output = self.classifier_layer(intial_out)

        return output




import cascor

def mainCascor():

    # exp_dir = "/home/grg/Research/DARPA-Pneumothorax/results/Diagnostic_Rules_Exps/tsm_LSU-Large-Dataset_Bio_Crop_upsampleVal_T1_C/reports_ml_tasks"
    exp_dir = "/home/grg/Research/DARPA-Pneumothorax/results/Diagnostic_Rules_Exps/tsm_LSU-Large-Dataset_Bio_Crop_upsampleVal_T1_C/reports_ml_gt_tasks"

    task = "lung-severity"
    # task = "diagnosis"
 
    exp_name = "Trial-1"
    reports_path = os.path.join(exp_dir, exp_name)
    utils.createDirIfDoesntExists(reports_path)

    net_size = 100 #100 #50 #25

    #Load data
    train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list = loadData(reports_path = exp_dir, task = task)

    class_names = ['score-0', 'score-1', 'score-2', 'score-3']
    # class_names = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others']
    num_classes = len(class_names)
    input_features = 38

    assert train_label_ft.shape == (len(gt_train_scores), input_features), "Error! Feature len mismatch."
    assert test_label_ft.shape == (len(gt_test_scores), input_features), "Error! Feature len mismatch."

    train_dataset = CustomDataset(filenames = train_video_list, inputs = train_label_ft, targets = gt_train_scores)
    test_dataset = CustomDataset(filenames = test_video_list, inputs = test_label_ft, targets = gt_test_scores)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print_results = ""

    #Initialize model

    x = train_label_ft
    # y = gt_train_scores[:, np.newaxis]
    y = torch.nn.functional.one_hot(torch.tensor(gt_train_scores)).numpy()

    # #Train
    # cascor.cascor_training(x, y)

    #Train initial output layer
    w, pred_prob = cascor.train_outputs(x, y)
    pred = pred_prob.argmax(1)

    # #Train hidden layer
    # neuron_w, neuron_value = cascor.train_hidden(x, y, pred_prob, debug=False)

    # #Combining Hidden & Output Neurons
    # x2 = np.concatenate((x, neuron_value), axis=1)

    # w2, pred_prob2 = cascor.train_outputs(x2, y)
    # pred2 = pred_prob2.argmax(1)

    pass


    #Calculate Scores 
    report_path = os.path.join(reports_path, f"classification_report_{task}.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Classification report")

    logger.log(f"Exp name: {exp_name}")

    model_results_dict = calScores(preds = pred, prob_preds = pred_prob, 
            targets = gt_train_scores, class_names = class_names, task = task, logger = logger, binary_cross_entropy=True)

    # model_results_dict = calScores(preds = pred2, prob_preds = pred_prob2, 
    #         targets = gt_train_scores, class_names = class_names, task = task, logger = logger, binary_cross_entropy=True)

    # utils.writeJson(model_results_dict, os.path.join(reports_path, f"classification_results_{task}.json"))

    # logger.close()

    pass



    pred_prob2, x2 = addHidden(x, y, pred_prob)
    pred2 = pred_prob2.argmax(1)

    model_results_dict = calScores(preds = pred2, prob_preds = pred_prob2, 
            targets = gt_train_scores, class_names = class_names, task = task, logger = logger, binary_cross_entropy=True)



    pred_prob3, x3 = addHidden(x2, y, pred_prob2)
    pred3 = pred_prob3.argmax(1)

    model_results_dict = calScores(preds = pred3, prob_preds = pred_prob3, 
            targets = gt_train_scores, class_names = class_names, task = task, logger = logger, binary_cross_entropy=True)




    pred_prob4, x4 = addHidden(x3, y, pred_prob3)
    pred4 = pred_prob4.argmax(1)

    model_results_dict = calScores(preds = pred4, prob_preds = pred_prob4, 
            targets = gt_train_scores, class_names = class_names, task = task, logger = logger, binary_cross_entropy=True)



    utils.writeJson(model_results_dict, os.path.join(reports_path, f"classification_results_{task}.json"))

    logger.close()

    pass




def addHidden(x, y, pred_prob):

    #Train hidden layer
    neuron_w, neuron_value = cascor.train_hidden(x, y, pred_prob, debug=False)

    #Combining Hidden & Output Neurons
    x2 = np.concatenate((x, neuron_value), axis=1)

    w2, pred_prob2 = cascor.train_outputs(x2, y)
    # pred2 = pred_prob2.argmax(1)

    return pred_prob2, x2



def mainCascorGRG():

    # exp_dir = "/home/grg/Research/DARPA-Pneumothorax/results/Diagnostic_Rules_Exps/tsm_LSU-Large-Dataset_Bio_Crop_upsampleVal_T1_C/reports_ml_tasks"
    exp_dir = "/home/grg/Research/DARPA-Pneumothorax/results/Diagnostic_Rules_Exps/tsm_LSU-Large-Dataset_Bio_Crop_upsampleVal_T1_C/reports_ml_gt_tasks"

    task = "lung-severity"
    # task = "diagnosis"
 
    exp_name = "Trial-1"
    reports_path = os.path.join(exp_dir, exp_name)
    utils.createDirIfDoesntExists(reports_path)

    net_size = 4 #1 #4
    
    SparsifyDuringTrain = False
    FreezePreviousLayers = True

    #Load data
    train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, train_video_list, test_video_list = loadData(reports_path = exp_dir, task = task)

    class_names = ['score-0', 'score-1', 'score-2', 'score-3']
    # class_names = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others']
    num_classes = len(class_names)
    input_features = 38

    assert train_label_ft.shape == (len(gt_train_scores), input_features), "Error! Feature len mismatch."
    assert test_label_ft.shape == (len(gt_test_scores), input_features), "Error! Feature len mismatch."

    train_dataset = CustomDataset(filenames = train_video_list, inputs = train_label_ft, targets = gt_train_scores)
    test_dataset = CustomDataset(filenames = test_video_list, inputs = test_label_ft, targets = gt_test_scores)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print_results = ""

    #Initialize model

    x = train_label_ft
    # y = gt_train_scores[:, np.newaxis]
    y = torch.nn.functional.one_hot(torch.tensor(gt_train_scores)).numpy()

    #Train

    classifier = nn.Linear(input_features, num_classes) #Output-layer
    print_results += f"\n\n base model \n\n"


    #Initialize optimizer

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier.parameters(), lr = 1e-3)

    #Train the model
    classifier, train_acc = train(classifier, train_dataloader, optimizer, criterion, num_classes, SparsifyDuringTrain)

    print_results += f"\n Train acc = {train_acc} \n"

    #Test the model
    test_acc = test(classifier, test_dataloader, criterion, task,
            class_names, reports_path, exp_name)

    print_results += f"\n Test acc = {test_acc} \n"

    ##[1st] Add hidden unit

    hidden = nn.Linear(input_features, net_size)
    

    hidden_criterion = covariance

    hidden_optimizer = torch.optim.Adam(hidden.parameters(), lr = 1e-3)


    hidden_train_dataset = CustomDataset(filenames = train_video_list, inputs = train_label_ft, targets = gt_train_scores)
    hidden_train_dataloader = DataLoader(hidden_train_dataset, batch_size=64, shuffle=False)
 
    #Test the model
    train_prob_preds, hidden_train_acc = test(classifier, hidden_train_dataloader, criterion, task,
            class_names, reports_path, exp_name, return_prob_preds = True)

    print_results += f"\n Hidden Train acc = {hidden_train_acc} \n"

    # Calculate the residuals for correlation
    err = torch.Tensor(y - train_prob_preds)
    # err_mean = torch.mean(err, axis=0)
    # err_corr = (err - err_mean)
    err_corr = err

    new_train_dataset = CustomDataset(filenames = train_video_list, inputs = train_label_ft, targets = err_corr)
    new_train_dataloader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)
 
    #Train the model
    hidden = trainHidden(hidden, new_train_dataloader, hidden_optimizer, hidden_criterion, num_classes, SparsifyDuringTrain)


    # print_results += f"\n Train acc = {train_acc} \n"

    new_classifier = cascorModel(input_features, num_classes, net_size, hidden, classifier)


    #Freeze previous model parameters
    if FreezePreviousLayers:
        for name, param in new_classifier.named_parameters():
            #Don't freeze the sequential classifier layer
            # if "new_" in name:
            if "new_" in name and 'prev_' not in name:
                print(f"Not freezing - {name}")
                continue

            param.requires_grad = False


    #Train the new model
    classifier = new_classifier
    print_results += f"\n\n 1st injection \n\n"


    def getModelPraramsToOptimize(model):
               
        params = filter(lambda p: p.requires_grad, model.parameters())

        return params

    optimizer = torch.optim.Adam(getModelPraramsToOptimize(classifier), lr = 1e-3)

    #Train the model
    classifier, train_acc = train(classifier, train_dataloader, optimizer, criterion, num_classes, SparsifyDuringTrain)

    print_results += f"\n Train acc = {train_acc} \n"

    #Test the model
    test_acc = test(classifier, test_dataloader, criterion, task,
            class_names, reports_path, exp_name)

    print_results += f"\n Test acc = {test_acc} \n"


    ##[2nd] Add hidden unit

    hidden = nn.Linear(input_features+net_size, net_size)
    

    hidden_criterion = covariance

    hidden_optimizer = torch.optim.Adam(hidden.parameters(), lr = 1e-3)


    hidden_train_dataset = CustomDataset(filenames = train_video_list, inputs = train_label_ft, targets = gt_train_scores)
    hidden_train_dataloader = DataLoader(hidden_train_dataset, batch_size=64, shuffle=False)
 
    #Test the model
    new_train_label_ft, train_prob_preds, hidden_train_acc = test(classifier, hidden_train_dataloader, criterion, task,
            class_names, reports_path, exp_name, return_prob_preds = True, return_features = True)

    print_results += f"\n Hidden Train acc = {hidden_train_acc} \n"

    # Calculate the residuals for correlation
    err = torch.Tensor(y - train_prob_preds)
    # err_mean = torch.mean(err, axis=0)
    # err_corr = (err - err_mean)
    err_corr = err


    new_train_dataset = CustomDataset(filenames = train_video_list, inputs = new_train_label_ft, targets = err_corr)
    new_train_dataloader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)
 
    #Train the model
    hidden = trainHidden(hidden, new_train_dataloader, hidden_optimizer, hidden_criterion, num_classes, SparsifyDuringTrain)


    # print_results += f"\n Train acc = {train_acc} \n"

    new_classifier = cascor2ndModel(num_classes, net_size, hidden, classifier)


    #Freeze previous model parameters
    if FreezePreviousLayers:
        for name, param in new_classifier.named_parameters():
            #Don't freeze the sequential classifier layer
            # if "new_" in name:
            if "new_" in name and 'prev_' not in name:
                print(f"Not freezing - {name}")
                continue

            param.requires_grad = False


    #Train the new model
    classifier = new_classifier
    print_results += f"\n\n 2nd injection \n\n"


    def getModelPraramsToOptimize(model):
               
        params = filter(lambda p: p.requires_grad, model.parameters())

        return params

    optimizer = torch.optim.Adam(getModelPraramsToOptimize(classifier), lr = 1e-3)

    #Train the model
    classifier, train_acc = train(classifier, train_dataloader, optimizer, criterion, num_classes, SparsifyDuringTrain)

    print_results += f"\n Train acc = {train_acc} \n"

    #Test the model
    test_acc = test(classifier, test_dataloader, criterion, task,
            class_names, reports_path, exp_name)

    print_results += f"\n Test acc = {test_acc} \n"


    print(print_results)
    pass



class cascorModel(nn.Module):

    def __init__(self, input_features, num_classes, net_size, hidden, classifier):
        super(cascorModel, self).__init__()

        # self.intial_layer = nn.Sequential(
        #         nn.Linear(input_features, net_size), nn.ReLU(), #Input-layer
        #     )
            
        self.hidden = hidden

        self.classifier_layer = classifier

        self.new_classifier_layer = nn.Linear(self.classifier_layer.in_features + net_size, num_classes) #Output-layer


        #Intialize with previous layer weights
        self.new_classifier_layer.weight[:, :input_features].data.copy_(self.classifier_layer.weight)
        self.new_classifier_layer.bias.data.copy_(self.classifier_layer.bias)

        #Check if the weights are correctly copied
        assert np.all((self.new_classifier_layer.weight[:, :input_features].cpu() == self.classifier_layer.weight.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."
        assert np.all((self.new_classifier_layer.bias.cpu() == self.classifier_layer.bias.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."


    def forward(self, inputs, return_features = False):

        hidden_out = self.hidden(inputs)

        if return_features:
            features = hidden_out.clone()

        output = self.new_classifier_layer(torch.cat((inputs, hidden_out), dim = 1))

        if return_features:
            return output, features

        return output


class cascor2ndModel(nn.Module):

    def __init__(self, num_classes, net_size, hidden, classifier):
        super(cascor2ndModel, self).__init__()

        # self.intial_layer = nn.Sequential(
        #         nn.Linear(input_features, net_size), nn.ReLU(), #Input-layer
        #     )
            
        self.hidden = hidden

        self.prev_classifier_layer = classifier

        input_features = self.prev_classifier_layer.new_classifier_layer.in_features
        self.new_classifier_layer = nn.Linear(input_features + net_size, num_classes) #Output-layer


        #Intialize with previous layer weights
        self.new_classifier_layer.weight[:, :input_features].data.copy_(self.prev_classifier_layer.new_classifier_layer.weight)
        self.new_classifier_layer.bias.data.copy_(self.prev_classifier_layer.new_classifier_layer.bias)

        #Check if the weights are correctly copied
        assert np.all((self.new_classifier_layer.weight[:, :input_features].cpu() == self.prev_classifier_layer.new_classifier_layer.weight.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."
        assert np.all((self.new_classifier_layer.bias.cpu() == self.prev_classifier_layer.new_classifier_layer.bias.cpu()).flatten().numpy()), "Error. Weights incorrectly copied."


    def forward(self, inputs, return_features = False):

        first_output, first_hidden = self.prev_classifier_layer(inputs, return_features = True)

        # hidden_out = self.hidden(inputs)
        hidden_out = self.hidden(torch.cat((inputs, first_hidden), dim = 1))

        if return_features:
            features = hidden_out.clone()

        # output = self.new_classifier_layer(torch.cat((inputs, hidden_out), dim = 1))
        output = self.new_classifier_layer(torch.cat((inputs, first_hidden, hidden_out), dim = 1))

        if return_features:
            return output, features

        return output



# Create a custom loss function (S)
def covariance(pred, target):
    pred_mean = torch.mean(pred, axis=0)
    target_mean = torch.mean(target, axis=0)
    # We want to try to maximize the absolute covariance, but quickprop will minimize its loss function
    # Therefore, we need to multiply by (-1) to guide the optimizer correctly
    # loss = -torch.sum(torch.abs(torch.sum((pred - pred_mean)*(target - target_mean), axis=0)), axis=0)
    # loss = -torch.mean(torch.abs(torch.mean((pred - pred_mean)*(target - target_mean), axis=0)), axis=0)
    loss = (1-(pred - pred_mean)*(target - target_mean)).mean()
    return loss


# Create a custom loss function (S)
def covarianceV0(pred, target):
    pred_mean = torch.mean(pred, axis=0)
    target_mean = torch.mean(target, axis=0)
    # We want to try to maximize the absolute covariance, but quickprop will minimize its loss function
    # Therefore, we need to multiply by (-1) to guide the optimizer correctly
    # loss = -torch.sum(torch.abs(torch.sum((pred - pred_mean)*(target - target_mean), axis=0)), axis=0)
    loss = -torch.mean(torch.abs(torch.mean((pred - pred_mean)*(target - target_mean), axis=0)), axis=0)
    return loss


# Create a custom loss function (S)
def covariance_org(pred, target):
    pred_mean = torch.mean(pred, axis=0)
    # We want to try to maximize the absolute covariance, but quickprop will minimize its loss function
    # Therefore, we need to multiply by (-1) to guide the optimizer correctly
    loss = -torch.sum(torch.abs(torch.sum((pred - pred_mean)*(target), axis=0)), axis=0)
    return loss



if __name__ == "__main__":
    print(f"Started...")
    

    # main()
    # mainCascor()
    mainCascorGRG()
    

    print("Finished!")