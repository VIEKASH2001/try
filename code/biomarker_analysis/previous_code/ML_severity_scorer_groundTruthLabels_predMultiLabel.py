from matplotlib import pyplot as plt
import numpy as np
import json

from sklearn import tree

if __name__ == '__main__':

   
    train_folds = ["B", "C"]
    
    dataset_split_dict_path = "/data1/datasets/LSU-Dataset/dataset_split_equi_class_R1.json"
    with open(dataset_split_dict_path, 'r') as json_file:
        dataset_split_dict = json.load(json_file)
            

    label_dict_path = "/data1/datasets/LSU-Dataset/user_label_gautamgare_12.json"
    # label_dict_path = 'user_label_tsmNet_1.json'
    with open(label_dict_path, 'r') as json_file:
        test_label_dict = json.load(json_file)

    # # label_dict_path = 'user_label_tsmNet_1.json'
    # with open('train_' + label_dict_path, 'r') as json_file:
    #     train_label_dict = json.load(json_file)
    train_label_dict = test_label_dict
    
    gt_label_dict_path = "/data1/datasets/LSU-Dataset/user_label_gautamgare_12.json"
    with open(gt_label_dict_path, 'r') as json_file:
        gt_label_dict = json.load(json_file)

    
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

    gt_test_scores = []
    [gt_test_scores.append(gt_label_dict[video]['lung-severity']) for video in test_video_list]
    assert np.array(gt_test_scores).shape == (len(test_video_list), 4)

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

    fig = plt.figure(figsize = (10,5), dpi=3000)
    # fig = plt.figure()
    # tree.plot_tree(clf) 
    tree.plot_tree(clf, 
                   feature_names=label_names,  
                   class_names=lung_severity_scores,
                   filled=True)
    plt.savefig("Decision Tree T5.png", bbox_inches='tight', dpi = 3000)
    plt.show()