
from sklearn.metrics import f1_score
import numpy as np
from .utilities import *
def merge_listinfo(info):
    if len(info) > 1:
        new_info = np.concatenate(info)
        return new_info
    return np.array(info)

def accuracy(out, labels):
    out = out > 0.5
    labels = np.squeeze(labels)
    correct = out==labels
    return 1.0*np.sum(correct)/float(labels.size)

class Meter():
    def __init__(self, clf_task=None):
        self.outputs = []
        self.intensity = []
        self.spectrum = []
        self.waveform = []
        self.label = []
        self.channel_name = []
        self.start_end = []
        self.clf_task = clf_task
        self.loss_ = []
    def add(self, spectrum, label, channel_name, start_end, intensity, waveform, output, loss=0):
        self.outputs.append(output)
        #self.spectrum.append(spectrum.numpy())
        self.label.append(label)
        self.channel_name.append(channel_name)
        self.start_end.append(start_end.numpy())
        #self.intensity.append(intensity.numpy())
        #self.waveform.append(waveform.numpy())
        self.loss_.append(loss)
    
    def update_loss(self, loss):
        self.loss_.append(loss)

    def update_outputs(self, output, label):
        self.outputs.append(output)
        self.label.append(label)

    def accuracy(self):
        outputs = merge_listinfo(self.outputs)
        labels = merge_listinfo(self.label)
        return accuracy(outputs, labels)
    def f1(self):
        outputs = merge_listinfo(self.outputs)
        labels = merge_listinfo(self.label)
        return f1_score(labels.squeeze(), (outputs>0.5).squeeze())

    def loss(self):
        return np.mean(merge_listinfo(self.loss_))

    def dump_pickle(self, filename):
        outputs = merge_listinfo(self.outputs)
        labels = merge_listinfo(self.label)
        spectrum = merge_listinfo(self.spectrum)
        channel_name = merge_listinfo(self.channel_name)
        start_end = merge_listinfo(self.start_end)
        intensity = merge_listinfo(self.intensity)
        waveform = merge_listinfo(self.waveform)
        
        res = {"outputs":outputs, 
                "labels":labels, 
                "spectrum":spectrum, 
                "channel_name":channel_name, 
                "start_end":start_end, 
                "intensity":intensity, 
                "waveform":waveform, 
                "task": self.clf_task}
 
        dump_pickle(filename, res)
    
    def dump_prediction(self, filename):
        outputs = merge_listinfo(self.outputs)
        labels = merge_listinfo(self.label)
        channel_name = merge_listinfo(self.channel_name)
        start_end = merge_listinfo(self.start_end)
        
        res = {"outputs":outputs, 
                "labels":labels, 
                "channel_name":channel_name, 
                "start_end":start_end, 
                "task": self.clf_task}
 
        dump_pickle(filename, res)

class TrainingMeter():
    def __init__(self, clf_task=None):
        self.training_acc = []
        self.validation_acc = []
        self.training_loss = []
        self.validation_loss = []
        self.training_f1 = []
        self.validation_f1 = []
        self.test_acc = []
        self.test_f1 = []
        self.clf_task = clf_task
    def add(
        self, 
        training_acc, training_loss ,
        validation_loss, validation_acc, 
        training_f1 = 0 , validation_f1 =0 ,
        test_acc=0, test_f1 = 0 
    ):
        self.training_acc.append(training_acc)
        self.training_loss.append(training_loss)
        self.training_f1.append(training_f1)
        self.validation_acc.append(validation_acc)
        self.validation_loss.append(validation_loss)
        self.validation_f1.append(validation_f1)
        self.test_acc.append(test_acc)
        self.test_f1.append(test_f1)

    def dump_pickle(self, filename):
        training_acc = np.array(self.training_acc)
        training_loss = np.array(self.training_loss)
        validation_acc = np.array(self.validation_acc)
        validation_loss = np.array(self.validation_loss)
        training_f1 = np.array(self.training_f1)
        validation_f1 = np.array(self.validation_f1)
        test_f1 = np.array(self.test_f1)
        test_acc = np.array(self.test_acc)

        res = {"training_acc":training_acc, 
                "training_loss":training_loss, 
                "validation_acc":validation_acc, 
                "validation_loss":validation_loss, 
                "training_f1":training_f1,
                "validation_f1":validation_f1, 
                "test_f1":test_f1, 
                "test_acc":test_acc, 
                "task": self.clf_task}
        dump_pickle(filename, res)
    
class InferenceStats:
    def __init__(self):
        self.start_end = []
        self.channel_names = []
        self.outputs_s = []
        self.outputs_a = []
        self.outputs_e = []
    
    def add(self, outputs_a, outputs_s, channel_names ,start_end):
        self.outputs_s.append(outputs_s)
        self.outputs_a.append(outputs_a)
        self.channel_names.append(channel_names)
        self.start_end.append(start_end)

    def export_cvs(self, fn):
        outputs_s = np.squeeze(np.concatenate(self.outputs_s))
        outputs_a = np.squeeze(np.concatenate(self.outputs_a))
        channel_names = np.concatenate(self.channel_names)
        start_end = np.concatenate(self.start_end)
        prediction_a = (outputs_a > 0.5).astype(int)
        prediction_s = (outputs_s > 0.5).astype(int)
        artifacts_index = np.where(prediction_a == 0)[0]
        prediction_s[artifacts_index] = -1
        result = np.concatenate((channel_names.reshape(-1, 1), start_end,prediction_a.reshape(-1, 1), prediction_s.reshape(-1, 1), outputs_a.reshape(-1, 1), outputs_s.reshape(-1, 1)), 1)
        pd.DataFrame(result).to_csv(fn,index=None ,header=["channel_names", "start", "end", "predictions_a", "predictions_s", "outputs_a", "outputs_s"])


class ClassificationStats(InferenceStats):
    def __init__(self):
        super(InferenceStats, self).__init__()
        self.outputs_e = []
    def add(self, outputs_a, outputs_s, outputs_e ,channel_names ,start_end):
        self.outputs_s.append(outputs_s)
        self.outputs_a.append(outputs_a)
        self.outputs_e.append(outputs_e)
        self.channel_names.append(channel_names)
        self.start_end.append(start_end)

    def export_cvs(self, fn):
        outputs_s = np.squeeze(np.concatenate(self.outputs_s))
        outputs_a = np.squeeze(np.concatenate(self.outputs_a))
        outputs_e = np.squeeze(np.concatenate(self.outputs_e))
        channel_names = np.concatenate(self.channel_names)
        start_end = np.concatenate(self.start_end)
        prediction_a = (outputs_a > 0.5).astype(int)
        prediction_s = (outputs_s > 0.5).astype(int)
        prediction_e = (outputs_e > 0.5).astype(int)
        artifacts_index = np.where(prediction_a == 0)[0]
        prediction_s[artifacts_index] = -1
        prediction_e[artifacts_index] = -1
        result = np.concatenate((channel_names.reshape(-1, 1), start_end,prediction_a.reshape(-1, 1), prediction_s.reshape(-1, 1), prediction_e.reshape(-1, 1), outputs_a.reshape(-1, 1), outputs_s.reshape(-1, 1), outputs_e.reshape(-1,1)), 1)
        pd.DataFrame(result).to_csv(fn,index=None ,header=["channel_names", "start", "end", "predictions_a", "predictions_s", "predictions_e", "outputs_a", "outputs_s", "outputs_e"])
