"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-01 22:43:12
 * @modify date 2021-12-01 17:58:11
 * @desc [description]
 """
 


import numpy as np
import torch 
import numpy as np
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import accuracy_score as accs



def cal_pred_accuracy(pred, truth):

    # equal = np.equal(pred, truth)
    # # acc = (equal == True).sum() / (pred.shape[0] * pred.shape[1])
    # # acc = (equal == True).sum() / pred.shape[0]
    # acc = (equal == True).mean()

    acc = np.equal(pred, truth).mean()

    return acc

def cal_IoU(pred, truth, labels, per_class = False):
    # pred = pred.reshape(-1)
    # truth = truth.reshape(-1)

    # IoU = jsc(truth, pred)
    # IoU = jsc(truth, pred, labels = labels, average = None if per_class else 'micro')
    IoU = jsc(truth, pred, labels = labels, average = None if per_class else 'macro')
    # acc = accs(truth, pred)
    return IoU


class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label = None):
    # def __init__(self, labels):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # # Mask-out value is ignored by default in the sklearn
        # # read sources to see how that was handled
        # # But sometimes all the elements in the groundtruth can
        # # be equal to ignore value which will cause the crush
        # # of scikit_learn.confusion_matrix(), this is why we check it here
        # if (ground_truth == self.ignore_label).all():
            
        #     return
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        
        if self.overall_confusion_matrix is not None:
            
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_current_mean_intersection_over_union(self):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        # union = ground_truth_set + predicted_set - intersection + 1
        union = ground_truth_set + predicted_set - intersection
        

        # intersection_over_union = intersection / union.astype(np.float32)
        intersection_over_union = intersection / (union.astype(np.float32)+1e-7)

        intersection_over_union_masked = np.ma.array(intersection_over_union, mask=False)
        if self.ignore_label is not None:
            intersection_over_union_masked.mask[self.ignore_label] = True
        mean_intersection_over_union = intersection_over_union_masked.mean() #Note - GRG : This is equivalent to scipy's "macro" average metric
        
        acc = intersection/(ground_truth_set+1e-7)

        # acc_masked = np.ma.array(acc, mask=False)
        # if self.ignore_label is not None:
        #     acc_masked.mask[self.ignore_label] = True
        # mean_acc = acc_masked.mean()


        intersection_masked = np.ma.array(intersection, mask=False)
        ground_truth_set_masked = np.ma.array(ground_truth_set, mask=False)
        if self.ignore_label is not None:
            intersection_masked.mask[self.ignore_label] = True
            ground_truth_set_masked.mask[self.ignore_label] = True
        mean_acc = intersection_masked.sum()/ground_truth_set_masked.sum()

        return mean_acc, acc, mean_intersection_over_union, intersection_over_union




class MetricEvaluation():
    
    def __init__(self, num_classes, ignore_label = None):

        self.label_class = np.arange(0, num_classes)

        self.IoU_obj = RunningConfusionMatrix(self.label_class, ignore_label)
    
    def evaluate(self, preds, targets):
        
        '''
        #Check for padded region - Note padded region index is 0, so find where label sum is 0
        labels = labels.squeeze()
        preds_1D = []
        targets_1D = []
        for idx, (output, label) in enumerate(zip(outputs.copy(), labels.copy())):
            
            label_sum = label.sum(axis=1)
            if label_sum.min() == 0:
                pad_idx = label_sum.argmin()
                #remove padded region
                output = output[:pad_idx, :]
                label = label[:pad_idx, :]

            preds_1D.extend(output.reshape(-1).tolist())
            targets_1D.extend(label.reshape(-1).tolist())

        assert len(preds_1D) == len(targets_1D), "Error! Size mismatch!"
        '''
        
        # preds_1D = preds.copy().reshape(-1)
        # targets_1D = targets.copy().reshape(-1)
        preds_1D = preds.reshape(-1)
        targets_1D = targets.reshape(-1)

        #----- mIoU -----#
        self.IoU_obj.update_matrix(preds_1D, targets_1D)


    def getMetrics(self):

        mean_acc, class_wise_mean_acc, mean_IoU, class_wise_mean_IoU = self.IoU_obj.compute_current_mean_intersection_over_union()

        return mean_acc, class_wise_mean_acc, mean_IoU, class_wise_mean_IoU



if __name__ == "__main__":

    targets = np.array([
                [[1,0,0,0], [1,0,0,0]],
                [[0,1,0,0], [0,1,0,0]],
                [[0,0,1,0], [0,0,1,0]],
                [[0,0,0,1], [0,0,0,1]],
                [[1,0,0,0], [1,0,0,0]],
                [[0,1,0,0], [0,1,0,0]],
                [[0,0,1,0], [0,0,1,0]],
                [[0,0,0,1], [0,0,0,1]],
                [[1,0,0,0], [1,0,0,0]],
                [[0,1,0,0], [0,1,0,0]],
                [[0,0,1,0], [0,0,1,0]],
                [[0,0,0,1], [0,0,0,1]],
            ])

    preds = np.array([
                [[0,1,0,0], [0,1,0,0]],
                [[0,0,1,0], [0,0,1,0]],
                [[0,0,0,1], [0,0,0,1]],
                [[1,0,0,0], [1,0,0,0]],
                [[1,0,0,0], [1,0,0,0]],
                [[0,1,0,0], [0,1,0,0]],
                [[0,0,1,0], [0,0,1,0]],
                [[0,0,0,1], [0,0,0,1]],
                [[1,0,0,0], [1,0,0,0]],
                [[0,1,0,0], [0,1,0,0]],
                [[0,0,0,1], [0,0,0,1]],
                [[0,0,1,0], [0,0,1,0]],
            ])

    metric = MetricEvaluation(4, 12)
    # metric = MetricEvaluation(4, 12, ignore_label=1)

    for p, t in zip(preds, targets):

        metric.evaluate(p, t)

    print(metric.getMetrics())
