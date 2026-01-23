from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
import json

class Averager:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def add(self, value):
        self.sum += value
        self.count += 1

    def get(self):
        return self.sum / self.count

class Recorder:
    def __init__(self, early_stop):
        self.early_stop = early_stop
        self.best_score = 0
        self.best_epoch = 0
        self.epoch = 0

    def update(self, score):
        self.epoch += 1
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.epoch
            return 'save'
        if self.epoch - self.best_epoch >= self.early_stop:
            return 'stop'
        else:
            return 'continue'

def acc_fpr1(y_true, y_score, sample_weight=None):
    fpr, tpr, thresholds = roc_curve(
        y_true, y_score, sample_weight=sample_weight, pos_label=1
    )
    idx = np.where(fpr <= 0.01)[0]
    return tpr[idx[-1]], thresholds[idx[-1]]

def metrics(y_true, y_score, is_binary):
    if is_binary:
        with open('./score.json', 'w') as f:
            json.dump({
                'y_true': y_true.tolist(),
                'y_score': y_score.tolist()
            }, f)
        results = dict()
        y_pred = y_score.round()
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['macf1'] = f1_score(y_true, y_pred, average='macro')
        results['precision'] = precision_score(y_true, y_pred, average='macro')
        results['recall'] = recall_score(y_true, y_pred, average='macro')
        results['auc_ovo'] = roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
        results['auc_ovr'] = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
        results['f1'] = f1_score(y_true, y_pred, average=None)
        results['acc_fpr1'] = acc_fpr1(y_true, y_score)
        results['sensitivity'] = recall_score(y_true, y_pred, pos_label=1)
        results['specificity'] = recall_score(y_true, y_pred, pos_label=0)

        return results
    else:
        results = dict()
        y_pred = y_score.argmax(dim=1)
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, y_pred, average='macro')
        results['precision'] = precision_score(y_true, y_pred, average='macro')
        results['recall'] = recall_score(y_true, y_pred, average='macro')
        try:
            if y_score.shape[1] == 2:
                y_score = y_score[:, 1]
            results['auc_ovo'] = roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
            results['auc_ovr'] = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
        except ValueError:
            results['auc_ovo'] = 0
            results['auc_ovr'] = 0
        print(f1_score(y_true, y_pred, average=None))
        return results
    
