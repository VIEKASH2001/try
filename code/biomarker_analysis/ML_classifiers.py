# Runtime optimization of sklearn uisng Intel - Ref: https://github.com/intel/scikit-learn-intelex
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()

# with config_context(target_offload="gpu:0"):
#     clustering = DBSCAN(eps=3, min_samples=2).fit(X)

from sklearn import tree, svm, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


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



