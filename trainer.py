from dataset_loader.data import *
from utils.metric import *
from utils.fix_random_seed import *
from utils.sliding_window_inference import *
from model.HDNet import *
from model.loss import *

import torch
import torch.utils.data as Data
from torch.optim import Adagrad
from tqdm import tqdm
import numpy as np
import os
import time

os.environ['CUDA_VISIBLE_DEVICES']="0"

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.warm_epoch = self.args.warm_epoch
        self.base_size = self.args.base_size
        self.step = self.args.step
        self.device = torch.device('cuda')
        self.start_epoch = 0
        self.best_iou = 0

        self.__build_dataloaders()
        self.__build_model()
        self.__build_optimizer()
        self.__build_loss()
        self.__build_metric()
        
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("GPU name:", torch.cuda.get_device_name(0))
        print("Model device:", next(self.model.parameters()).device)
    
    def __build_dataloaders(self):
        trainset = Segmentation_Dataset_train(self.args, mode='train')
        valset = Segmentation_Dataset_val(self.args, mode='val')

        self.train_loader = Data.DataLoader(trainset, self.args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

    def __build_model(self):
        model = HDNet(3)
        if self.args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use ' + str(torch.cuda.device_count()) + ' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(self.device)
        self.model = model

    def __build_optimizer(self):
        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 25, eta_min=0.045, last_epoch=-1)

    def __build_loss(self):
        self.loss_fun = MultiScaleSLSIoULoss(base_loss=SLSIoULoss(), downsample=nn.MaxPool2d(2,2))
    
    def __build_metric(self):
        self.miou = mIoU(1)
        self.roc  = ROCMetric(1, 10)
        self.train_loss_curve = []
        self.val_iou_curve = []
        self.roc_curve_per_epoch = []

    def load_checkpoint(self):
        if self.args.if_checkpoint:
            checkpoint = torch.load(self.args.checkpoint_path)
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']+1
            self.best_iou = checkpoint['iou']
            self.save_folder = os.path.dirname(self.args.checkpoint_path)
        else:
            self.start_epoch = 0
            self.best_iou = 0
            self.save_folder = os.path.join(self.args.save_dir, 'HDNet-%s'%(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))))
            if not os.path.exists(self.save_folder):
                os.mkdir(self.save_folder)

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        loss_all = AverageMeter()
        tag = epoch>self.warm_epoch
        for i, (data, mask) in enumerate(tbar):
  
            data = data.to(self.device)
            labels = mask.to(self.device)

            masks, pred = self.model(data, tag)

            loss = self.loss_fun(pred, masks, labels, self.warm_epoch, epoch)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
            loss_all.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, loss_all.avg))
        self.scheduler.step()
        self.train_loss_curve.append(loss_all.avg)

    def validate(self, epoch):
        self.model.eval()
        self.miou.reset()
        self.roc.reset()
        tbar = tqdm(self.val_loader)
        tag = epoch>self.warm_epoch
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):

                data = data.to(self.device)
                mask = mask.to(self.device)

                pred = sliding_window_inference(self.model, data, self.base_size, self.step, self.device, tag)

                self.miou.update(pred, mask)
                self.roc.update(pred, mask)

                _, mean_iou = self.miou.get()
                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_iou))
            _, mean_iou = self.miou.get()
            ture_positive_rate, false_positive_rate, recall, precision = self.roc.get()

            self.val_iou_curve.append(mean_iou)
            self.roc_curve_per_epoch.append({'epoch': epoch,
                                             'tpr': ture_positive_rate.copy(),
                                             'fpr': false_positive_rate.copy(),
                                             'recall': recall.copy(),
                                             'precision': precision.copy()})

            self.__save_best_weight(epoch, mean_iou, recall, precision)
            self.__save_epoch_log(epoch, mean_iou)
            self.__save_curves()
    
    def __save_epoch_log(self, epoch, mean_iou, recall, precision):
        log_path = os.path.join(self.save_folder, 'log.txt')
        with open(log_path, 'a') as f:
            f.write('{} - {:04d}\t - IoU {:.4f}\t - Recall {:.4f}\t - Precision {:.4f}\n'.format(
                time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                epoch,
                mean_iou,
                recall,
                precision))
            
        all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
        torch.save(all_states, os.path.join(self.save_folder, 'checkpoint.pkl'))

    def __save_best_weight(self, epoch, mean_iou, recall, precision):
        if mean_iou > self.best_iou:
            self.best_iou = mean_iou
            metric_path = os.path.join(self.save_folder, 'metric.log')
            with open(metric_path, 'a') as f:
                f.write('{} - {:04d}\t - IoU {:.4f}\t - Recall {:.4f}\t - Precision {:.4f}\n'.format(
                    time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                    epoch,
                    self.best_iou,
                    recall,
                    precision))
            torch.save(self.model.state_dict(), os.path.join(self.save_folder, 'weight.pkl'))
    
    def __save_curves(self):
        torch.save({'train_loss': self.train_loss_curve, 
                    'val_iou': self.val_iou_curve},
                    os.path.join(self.save_folder, 'learning_curve.pth'))
        np.savez(os.path.join(self.save_folder, 'roc_curve.npz'),
                 roc=self.roc_curve_per_epoch)
        