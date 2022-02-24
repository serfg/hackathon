import os
import sys
import gc; gc.enable()
#import warnings; warnings.filterwarnings("ignore")

#import pickle
from tqdm import tqdm

import numpy as np
#import pandas as pd
#import seaborn as sns
from tabulate import tabulate
from matplotlib import pyplot as plt
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from torch.utils.data import Dataset, DataLoader

#import torch_geometric
#from torch_geometric.nn import GCNConv
#from torch_geometric.nn import global_max_pool
#from torch_geometric.loader import DataLoader

#from openbabel import pybel
#pybel.ob.obErrorLog.SetOutputLevel(0)

#from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score


class Trainer(object):
    def __init__(self, model, opt, scheduler, train_loader, val_loader, num_epochs,
                 weight=10.0, step='step', backup_by='adaptive_f1',
                 logs_path='./logs', path_to_save='./ckpt', exp_name='test', verbose=2):
        self.model = model
        self.opt = opt
        self.scheduler, self.step = scheduler, step
        
        self.train_loader, self.val_loader = train_loader, val_loader
        self.weight = torch.FloatTensor([weight])
        
        self.train_losses, self.val_losses = [], []
        self.train_aucs, self.val_aucs = [], []
        
        self.adaptive_f1s = []
        self.f1_curve, self.thr = None, None
        
        self.logs_path, self.path_to_save, self.exp_name = logs_path, path_to_save, exp_name
        os.makedirs(self.path_to_save, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.logs_path, self.exp_name))
        
        self.num_epochs, self.backup_by = num_epochs, backup_by
        self.verbose, self.epoch, self.best_epoch = verbose, 0, 0
        
        
    def save_checkpoint(self):
        name = self.exp_name + '_' + str(self.epoch) + '.pth'
        torch.save(self.model.state_dict(), os.path.join(self.path_to_save, name))
        
        
    @staticmethod    
    def adaptive_f1(logits, targets, delta=0.001):
        def _f1(outputs, targets, thr):
            return f1_score(targets, outputs > thr)

        outputs = (1.0 / (1.0 + np.exp(-logits)))
        thrs, f1s = np.arange(0.0, 1.0 + delta, delta), []
        for thr in thrs:
            f1s.append(_f1(outputs, targets, thr))

        index = np.argmax(f1s)
        return thrs[index], f1s[index], thrs, f1s
    
    
    @staticmethod
    def roc_auc(logits, targets):
        return roc_auc_score(targets, (1.0 / (1.0 + np.exp(-logits))))
    
        
    def iterate_loader(self, loader, train=False):
        self.model.train(train)
        device = self.model.parameters().__iter__().__next__().device
        logits, targets, losses = [], [], []
        
        if self.verbose > 1:
            generator = tqdm(loader, desc=str(self.epoch + 1) + ', ' + ('train' if train else '  val'))
        else:
            generator = loader
        with torch.set_grad_enabled(train):
                
            for batch in generator:
                logit = self.model.forward(batch.to(device))

                logits.extend(logit.detach().cpu().numpy())            
                targets.extend(batch.y.cpu().numpy())

                loss = F.binary_cross_entropy_with_logits(logit, batch.y.to(device).float(),
                                                          reduction='none',
                                                          pos_weight=self.weight.to(device))
                losses.extend(loss.detach().cpu().numpy())

                if train:
                    self.opt.zero_grad()
                    loss.mean().backward()
                    self.opt.step()
                    if self.scheduler is not None and self.step == 'step':
                        self.scheduler.step()

        if train and self.scheduler is not None and self.step != 'step':
            self.scheduler.step() 
        logits, targets, losses = np.asarray(logits), np.asarray(targets), np.asarray(losses)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        return logits, targets, losses
    
    
    def run_epoch(self):
        train_logits, train_targets, train_losses = self.iterate_loader(self.train_loader, train=True)
        val_logits, val_targets, val_losses = self.iterate_loader(self.val_loader, train=False)
        
        train_auc = self.roc_auc(train_logits, train_targets)
        val_auc = self.roc_auc(val_logits, val_targets)
        
        self.train_losses.append(np.mean(train_losses))
        self.val_losses.append(np.mean(val_losses))
        
        self.train_aucs.append(train_auc)
        self.val_aucs.append(val_auc)
        
        thr, f1, thrs, f1s = self.adaptive_f1(val_logits, val_targets)
        self.adaptive_f1s.append(f1)
        # self.f1_curve, self.thr = np.asarray([thrs, f1s]), thr
       
        self.epoch += 1
        
        self.writer.add_scalar('loss/train', self.train_losses[-1], self.epoch)
        self.writer.add_scalar('loss/val', self.val_losses[-1], self.epoch)
        
        self.writer.add_scalar('auc/train', self.train_aucs[-1], self.epoch)
        self.writer.add_scalar('auc/val', self.val_aucs[-1], self.epoch)
        
        self.writer.add_scalar('adaptive_f1/val', self.adaptive_f1s[-1], self.epoch)
        self.writer.flush()
        
        if self.verbose > 1:
            print(tabulate([['train', self.train_losses[-1], self.train_aucs[-1], ''],
                            ['  val', self.val_losses[-1], self.val_aucs[-1], self.adaptive_f1s[-1]]],
                           headers=['', 'loss', 'auc', 'adaptive f1'],
                           tablefmt='simple', floatfmt='.6f'))
        if self.verbose > 2:
            self.plot_result()
        
        if self.verbose > 3:
            self.plot_f1_curve()
            
        if self.val_losses[-1] != min(self.val_losses) and self.backup_by == 'loss':
            return        
        if self.val_aucs[-1] != max(self.val_aucs) and self.backup_by == 'auc':
            return
        if self.adaptive_f1s[-1] != max(self.adaptive_f1s) and self.backup_by == 'adaptive_f1':
            return
        
        self.save_checkpoint()
        self.best_epoch = self.epoch
        self.f1_curve, self.thr = np.asarray([thrs, f1s]), thr
        if self.verbose > 1:
            print('save checkpoint')
        
        
    def run(self):
        generator = range(self.num_epochs) if self.verbose > 1 else tqdm(range(self.num_epochs))
        for _ in generator:
            self.run_epoch()
        self.writer.close() 
    
    
    def plot_result(self):
        if self.epoch < 2:
            return
        
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, self.epoch + 1), self.train_losses, label='train', color='orange')
        plt.plot(np.arange(1, self.epoch + 1), self.val_losses, label='val', color='red')
        plt.xlabel('epoch')
        plt.title('WBCEloss')
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, self.epoch + 1), self.train_aucs, label='train', color='orange')
        plt.plot(np.arange(1, self.epoch + 1), self.val_aucs, label='val', color='red')
        plt.xlabel('epoch')
        plt.title('ROC AUC')
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, self.epoch + 1), self.adaptive_f1s, label='val', color='red')
        plt.xlabel('epoch')
        plt.title('Adaptive F1')
        plt.legend()
        plt.grid()
        plt.show()
        
        
    def plot_f1_curve(self):
        if self.f1_curve is None:
            return
        
        x, y = self.f1_curve[0], self.f1_curve[1]
            
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='val')
        plt.xlabel('thr')
        plt.title('F1')
        plt.grid()
        plt.show()
        
        
def inference(model, loader, y=False):
    model.train(False)
    device = model.parameters().__iter__().__next__().device
    logits, targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            logit = model.forward(batch.to(device))
            logits.extend(logit.detach().cpu().numpy())
            
            if y:
                targets.extend(batch.y.cpu().numpy())

    if not y:
        return np.asarray(logits)
    return np.asarray(logits), np.asarray(targets)