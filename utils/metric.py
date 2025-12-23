import  numpy as np
import torch.nn as nn
import torch
from skimage import measure

class ROCMetric():
    """
    Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins): 
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_pos, i_fp, i_neg,i_class_pos = self.__cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)

        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr   = np.zeros([self.bins+1])
        self.pos_arr  = np.zeros([self.bins+1])
        self.fp_arr   = np.zeros([self.bins+1])
        self.neg_arr  = np.zeros([self.bins+1])
        self.class_pos= np.zeros([self.bins+1])

    def __cal_tp_pos_fp_neg(self, output, target, nclass, score_thresh):
        predict = (torch.sigmoid(output) > score_thresh).float()
        if len(target.shape) == 3:
            target = np.expand_dims(target.float(), axis=1)
        elif len(target.shape) == 4:
            target = target.float()
        else:
            raise ValueError("Unknown target dimension")

        tp = torch.sum(predict * ((predict == target).float())).item()
        fp = torch.sum((predict * ((predict != target).float()))).item()
        tn = torch.sum(((1 - predict) * ((predict == target).float()))).item()
        fn = torch.sum((((predict != target).float()) * (1 - predict))).item()
        pos = tp + fn
        neg = fp + tn
        class_pos = tp + fp

        return tp, pos, fp, neg, class_pos

class mIoU():
    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        correct, labeled = self.__batch_pix_accuracy(preds, labels)
        inter, union = self.__batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def __batch_pix_accuracy(self, output, target):

        if len(target.shape) == 3:
            target = np.expand_dims(target.float(), axis=1)
        elif len(target.shape) == 4:
            target = target.float()
        else:
            raise ValueError("Unknown target dimension")

        assert output.shape == target.shape, "Predict and Label Shape Don't Match"
        predict = (output > 0).float()
        pixel_labeled = (target > 0).float().sum()
        pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()

        assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
        return pixel_correct, pixel_labeled


    def __batch_intersection_union(self, output, target, nclass):

        mini = 1
        maxi = 1
        nbins = 1
        predict = (output > 0).float()
        if len(target.shape) == 3:
            target = np.expand_dims(target.float(), axis=1)
        elif len(target.shape) == 4:
            target = target.float()
        else:
            raise ValueError("Unknown target dimension")
        intersection = predict * ((predict == target).float())

        area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
        area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
        area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
        area_union     = area_pred + area_lab - area_inter

        assert (area_inter <= area_union).all(), \
            "Error: Intersection area should be smaller than Union area"
        return area_inter, area_union

