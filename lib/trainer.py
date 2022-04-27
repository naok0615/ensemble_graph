# coding: utf-8

import pdb
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from lib import utils


class ClassificationTrainer():
    def __init__(self, args):
        self.args = args
        pass
    
    def to_cuda(self, input_, target):
        input_ = input_.cuda(self.args.gpu.id)
        target = target.cuda(self.args.gpu.id)
        return input_, target
    
    def compute_outputs(self, models, input_, target):
        outputs = [model(input_) for model in models]
        return outputs
        
    def post_forward(self, models, outputs):
        for id_, model in enumerate(models):
            if hasattr(model, "post_forward"):
                outputs[id_] = model.post_forward(outputs)
        return outputs
        
    def calc_accuracy(self, output, target):
        return utils.accuracy(output, target, topk=(1,))
    
    def measure(self, output, target):
        return calc_accuracy(output, target)
    
    def update_meter(self, loss, acc, loss_meter, top1_meter, batch_size):
        # record loss and accuracy 
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc[0].item(), batch_size)
        return
    
    def write_log(self, logs, loss_meters, top1_meters, epoch, mode="train"):
        for log, loss_meter, top1_meter in zip(logs.net, loss_meters, top1_meters):
            log["epoch_log"][epoch][mode+"_loss"] = loss_meter.avg
            log["epoch_log"][epoch][mode+"_accuracy"] = top1_meter.avg
        return
    
    def train_on_batch(self, input_, target, models, criterions, optimizers, 
                       logs, loss_meters, top1_meters, **kwargs):
                
        outputs = self.compute_outputs(models, input_, target)
        outputs = self.attention_masked(outputs, 0.5, **kwargs)
        outputs = self.post_forward(models, outputs)
                
        losses = []
        for model_id, (criterion,
                       optimizer,
                       log,
                       loss_meter,
                       top1_meter) in enumerate(zip(criterions,
                                                    optimizers,
                                                    logs.net,
                                                    loss_meters,
                                                    top1_meters)):
            
            loss = criterion(model_id, outputs, target, log, **kwargs)
            losses += [loss]
            
            acc = self.calc_accuracy(outputs[model_id], target)
            
            self.update_meter(loss, acc, loss_meter, top1_meter, batch_size=input_.size(0))
        
        
        # initialize gradient
        for optimizer in optimizers:
            optimizer.zero_grad()
        # exclude loss if it equal 0
        update_idxs = [id_ for id_, loss in enumerate(losses) if loss != 0]
        # compute gradient
        for id_ in update_idxs:
            losses[id_].backward(retain_graph=True)
        # update paramaters
        for id_ in update_idxs:
            optimizers[id_].step()
        
        return
    
    def train_on_dataset(self, data_loader, models, criterions, optimizers, epoch, logs, **kwargs):
        """
            train on dataset for one epoch
        """
        
        loss_meters = [utils.AverageMeter() for i in range(len(models))]
        top1_meters = [utils.AverageMeter() for i in range(len(models))]
        
        for model in models:
            model.train()
        
        for i, (input_, target) in enumerate(tqdm(data_loader)):          
            input_, target = self.to_cuda(input_, target)
            self.train_on_batch(input_, target, models, criterions, optimizers, 
                                logs, loss_meters, top1_meters, **kwargs)
        
        self.write_log(logs, loss_meters, top1_meters, epoch, mode="train")
        
        return logs
    
    def validate_on_batch(self, input_, target, models, criterions, logs, loss_meters, top1_meters, **kwargs):
        outputs = self.compute_outputs(models, input_, target)
        
        outputs = self.attention_masked(outputs, 1, **kwargs)
        
        outputs = self.post_forward(models, outputs)        
        for model_id, (criterion,
                       log,
                       loss_meter,
                       top1_meter) in enumerate(zip(criterions,
                                                    logs.net,
                                                    loss_meters,
                                                    top1_meters)):
            
            loss = criterion(model_id, outputs, target, log=None, **kwargs)
            
            acc = self.calc_accuracy(outputs[model_id], target)
            
            self.update_meter(loss, acc, loss_meter, top1_meter, batch_size=input_.size(0))
        
        return
    
    def validate_on_dataset(self, data_loader, models, criterions, epoch, logs, **kwargs):
        """
            validate on dataset
        """
        
        loss_meters = [utils.AverageMeter() for i in range(len(models))]
        top1_meters = [utils.AverageMeter() for i in range(len(models))]
        
        for model in models:
            model.eval()
        
        for i, (input_, target) in enumerate(tqdm(data_loader)):
            input_, target = self.to_cuda(input_, target)            
            self.validate_on_batch(input_, target, models, criterions,
                                   logs, loss_meters, top1_meters, **kwargs)
        
        self.write_log(logs, loss_meters, top1_meters, epoch, mode="test")
        
        return logs

    def train(self, nets, criterions, optimizers,
              train_loader, test_loader, logs=None, **kwargs):
        import time
        import os
        
        print("manual seed : %d" % self.args.manualSeed)
        
        for epoch in range(self.args.trainer.start_epoch, self.args.trainer.epochs+1):
            print("epoch %d" % epoch)
            
            start_time = time.time()
            
            for optimizer, model_args in zip(optimizers, self.args.models):
                utils.adjust_learning_rate(optimizer,
                                           epoch,
                                           model_args.optim.gammas, 
                                           model_args.optim.schedule,
                                           model_args.optim.args.lr)
            
            kwargs = {}  if kwargs is None else kwargs
            kwargs.update({
                "_trainer": self,
                "_train_loader": train_loader,
                "_test_loader": test_loader,
                "_nets": nets, 
                "_criterions": criterions, 
                "_optimizers": optimizers, 
                "_epoch": epoch,
                "_logs": logs,
                "_args" :self.args,
            })
            
            # train for one epoch
            self.train_on_dataset(train_loader, nets, criterions, optimizers, epoch, logs, **kwargs)
            # evaluate on validation set
            self.validate_on_dataset(test_loader, nets, criterions, epoch, logs, **kwargs)
            
            # print log
            for i, log in enumerate(logs.net):
                print("  net{0}    loss :train={1:.3f}, test={2:.3f}    acc :train={3:.3f}, test ={4:.3f}".format(
                    i,
                    log["epoch_log"][epoch]["train_loss"],
                    log["epoch_log"][epoch]["test_loss"],
                    log["epoch_log"][epoch]["train_accuracy"],
                    log["epoch_log"][epoch]["test_accuracy"]))
                        
            if epoch % self.args.trainer.saving_interval == 0:
                ckpt_dir = os.path.join(self.args.trainer.base_dir, "checkpoint")
                utils.save_checkpoint(nets,optimizers, epoch, ckpt_dir)
            
            logs.save(self.args.trainer.base_dir + r"log/")
            
            elapsed_time = time.time() - start_time
            print ("  elapsed_time:{0:.3f}[sec]".format(elapsed_time))
            
            if "_callback" in kwargs:
                kwargs["_callback"](**kwargs)
        
        return
    
    # for attention loss  -------------------------------------------------------------------------------------
    def map2vec(self, attention):
        return attention.view(attention.size(0), -1)
    
    def map2vec_l2_normalization(self, attention):
        return F.normalize(self.map2vec(attention))
    
    def top_att_id(self, attention):
        data_id_list = []
        
        for data_id in range(len(attention)):
            # ABN_attention     : torch.Size([8, 1, 14, 14])
            # AttentionTransfer : torch.Size([8, 7, 7])
            top_id = attention[data_id][0].argmax().item()
            y_id = np.trunc(top_id/14)
            x_id = top_id - y_id*14
            
            set_list = []
            
            # 11*11 ---------------------------------
            if   y_id<5:
                set_y_1 = 0
                set_y_2 = 11
            elif y_id>8:
                set_y_1 = 3
                set_y_2 = 14
            else:
                set_y_1 = int(y_id)-5
                set_y_2 = int(y_id)+6

            if   x_id<5:
                set_x_1 = 0
                set_x_2 = 11
            elif x_id>8:
                set_x_1 = 3
                set_x_2 = 14
            else:
                set_x_1 = int(x_id)-5
                set_x_2 = int(x_id)+6
            set_list.append([list(range(set_y_1, set_y_2)), set_x_1, set_x_2])
            # 11*11 ---------------------------------
            # 7*7 ---------------------------------
            if   y_id<3:
                set_y_1 = 0
                set_y_2 = 7
            elif y_id>10:
                set_y_1 = 7
                set_y_2 = 14
            else:
                set_y_1 = int(y_id)-3
                set_y_2 = int(y_id)+4

            if   x_id<3:
                set_x_1 = 0
                set_x_2 = 7
            elif x_id>10:
                set_x_1 = 7
                set_x_2 = 14
            else:
                set_x_1 = int(x_id)-3
                set_x_2 = int(x_id)+4
            set_list.append([list(range(set_y_1, set_y_2)), set_x_1, set_x_2])
            # 7*7 ---------------------------------
            # 3*3 ---------------------------------
            if   y_id<1:
                set_y_1 = 0
                set_y_2 = 3
            elif y_id>12:
                set_y_1 = 11
                set_y_2 = 14
            else:
                set_y_1 = int(y_id)-1
                set_y_2 = int(y_id)+2

            if   x_id<1:
                set_x_1 = 0
                set_x_2 = 3
            elif x_id>12:
                set_x_1 = 11
                set_x_2 = 14
            else:
                set_x_1 = int(x_id)-1
                set_x_2 = int(x_id)+2
            set_list.append([list(range(set_y_1, set_y_2)), set_x_1, set_x_2])
            # 3*3 ---------------------------------
            
            data_id_list.append(set_list)
            
        return data_id_list
    
    def attention_masked(self, output, mask_per, **kwargs):
        for k in range(len(output)-1):  # number of network_node - ensemble_node
            output[k+1][2] = [self.map2vec(output[k+1][2]), self.top_att_id(output[k+1][2])]

        return output
