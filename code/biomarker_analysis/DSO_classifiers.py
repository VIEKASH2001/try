
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

