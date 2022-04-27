# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


# Total Loss Function
class TotalLoss(nn.modules.loss._Loss):
    def __init__(self, loss_funcs):
        super(TotalLoss, self).__init__()
        self.loss_funcs = loss_funcs
        self.ite = 1

    def forward(self, model_id, outputs, label_id, log=None, **kwargs):
        losses = []
        target_output = outputs[model_id]
        for source_id, (source_output, loss_func) in enumerate(zip(outputs, self.loss_funcs)):
            kwargs["_source_id"] = source_id
            kwargs["_target_id"] = model_id
            losses += [loss_func(target_output,
                                 source_output,
                                 label_id, 
                                 log,
                                 **kwargs)]
        total_loss = torch.stack(losses).sum()
        
        if log is not None:
            for source_id, (func, loss) in enumerate(zip(self.loss_funcs, losses)):
                log["ite_log"][self.ite][f"{model_id:02}_{source_id:02}_"+func.__class__.__name__] = float(loss)
            log["ite_log"][self.ite][f"{model_id:02}_loss"] = float(total_loss)
            self.ite += 1
        return total_loss


# Loss Functioin

# Base Class
class _LossBase(nn.modules.loss._Loss):
    def __init__(self, args):
        super(_LossBase, self).__init__()
        self.args = args
        self.ite = 1

    def forward(self, target_output, source_output, label_id, log=None, **kwargs):
        return 


# Independent Loss
class IndependentLoss(_LossBase):
    def __init__(self, args):
        super(IndependentLoss, self).__init__(args)
        self.gate = globals()[args.gate.name](self, args.gate.args)
        return

    def forward(self, target_output, source_output, label_id, log=None, **kwargs):
        loss_per_sample = F.cross_entropy(target_output[0], label_id, reduction='none')
        
        if target_output[1] is not None:
            loss_per_sample += F.cross_entropy(target_output[1], label_id, reduction='none')
        
        kwargs["_student_logits"] = target_output[0]       
        identity = torch.eye(target_output[0].shape[-1], device=target_output[0].device)
        onehot_label = identity[label_id]
        kwargs["_teacher_logits"] = onehot_label
        kwargs["_label_id"] = label_id
        
        hard_loss = self.gate.f(loss_per_sample, log, **kwargs)        
        
        loss = hard_loss * self.args.loss_weight
        
        if log is not None:
            self.ite = self.ite + 1
            
        return loss


# Cross entropy loss
class CELoss(_LossBase):
    def __init__(self, args):
        super(CELoss, self).__init__(args)
        self.T = args.T
        self.loss_weight = args.loss_weight
        self.gate = globals()[args.gate.name](self, args.gate.args)
        return

    def forward(self, target_output, source_output, label_id, log=None, **kwargs): 
        student_logits = target_output[0]
        teacher_logits = source_output[0].detach()
        
        soft_loss_per_sample = self.cross_entropy(student_logits, teacher_logits, T=self.T)

        kwargs["_student_logits"] = student_logits
        kwargs["_teacher_logits"] = teacher_logits
        kwargs["_label_id"] = label_id
        
        soft_loss = self.gate.f(soft_loss_per_sample, log, **kwargs)
                
        loss = soft_loss * (self.T**2) * (self.loss_weight)
        
        if log is not None:
            self.ite = self.ite + 1
        
        return loss
    
    def cross_entropy(self, student_logits, teacher_logits, T):
        student_log_softmax = F.log_softmax(student_logits/T, dim=1)
        teacher_softmax     = F.softmax(teacher_logits/T, dim=1)
        ce = -(teacher_softmax * student_log_softmax).sum(dim=1)
        return ce
    
    
# KL Distillation Loss
class KLLoss(_LossBase):
    def __init__(self, args):
        super(KLLoss, self).__init__(args)
        self.T = args.T
        self.gate = globals()[args.gate.name](self, args.gate.args)
        self.soft_loss_positive = args.soft_loss_positive
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        return

    def forward(self, target_output, source_output, label_id, log=None, **kwargs): 
        student_logits = target_output[0]
        teacher_logits = source_output[0].detach()
        
        student_softmax = F.softmax(student_logits/self.T, dim=1)
        teacher_softmax = F.softmax(teacher_logits/self.T, dim=1)
        
        if   self.soft_loss_positive == 1:
            soft_loss_per_sample = self.kl_divergence(student_softmax, teacher_softmax, log=log)
        elif self.soft_loss_positive == 0:
            soft_loss_per_sample = self.cos_sim(student_softmax, teacher_softmax)
        
        kwargs["_student_logits"] = student_logits
        kwargs["_teacher_logits"] = teacher_logits
        kwargs["_label_id"] = label_id
        
        soft_loss = self.gate.f(soft_loss_per_sample, log, **kwargs)
                
        loss = soft_loss * (self.T**2)
        
        if log is not None:
            self.ite = self.ite + 1
        
        return loss
    
    def kl_divergence(self, student, teacher, log=None):
        kl = teacher * torch.log((teacher / (student+1e-10)) + 1e-10)
        kl = kl.sum(dim=1)
        loss = kl
        return loss


# Attention Loss
class AttLoss(_LossBase):
    def __init__(self, args):
        super(AttLoss, self).__init__(args)
        self.loss_weight = args.loss_weight
        self.gate = globals()[args.gate.name](self, args.gate.args)
        self.att_loss_positive  = args.att_loss_positive
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        return

    def forward(self, target_output, source_output, label_id, log=None, **kwargs):
        # Attention loss
        student_att, student_att_top_index = target_output[2]
        teacher_att, teacher_att_top_index = source_output[2]
        teacher_att = teacher_att.detach()
        
        # ------------------------------------------------------------------------------------------------
        # 5*5
        student_att_top_5 = torch.zeros(len(student_att), 5*5, device=student_att.device)
        teacher_att_top_5 = torch.zeros(len(teacher_att), 5*5, device=teacher_att.device)
        # 3*3
        student_att_top_3 = torch.zeros(len(student_att), 3*3, device=student_att.device)
        teacher_att_top_3 = torch.zeros(len(teacher_att), 3*3, device=teacher_att.device)
        
        for i in range(len(student_att)):
            # 5*5 -----------------------------------------------------------------------------------
            y_id, x_1, x_2 = teacher_att_top_index[i][0]
            student_att_top_5[i] = torch.cat((student_att[i][ x_1+7*y_id[0] : x_2+7*y_id[0] ],
                                              student_att[i][ x_1+7*y_id[1] : x_2+7*y_id[1] ],
                                              student_att[i][ x_1+7*y_id[2] : x_2+7*y_id[2] ],
                                              student_att[i][ x_1+7*y_id[3] : x_2+7*y_id[3] ],
                                              student_att[i][ x_1+7*y_id[4] : x_2+7*y_id[4] ],
                                             ))
            teacher_att_top_5[i] = torch.cat((teacher_att[i][ x_1+7*y_id[0] : x_2+7*y_id[0] ],
                                              teacher_att[i][ x_1+7*y_id[1] : x_2+7*y_id[1] ],
                                              teacher_att[i][ x_1+7*y_id[2] : x_2+7*y_id[2] ],
                                              teacher_att[i][ x_1+7*y_id[3] : x_2+7*y_id[3] ],
                                              teacher_att[i][ x_1+7*y_id[4] : x_2+7*y_id[4] ],
                                             ))
            # 3*3 -----------------------------------------------------------------------------------
            y_id, x_1, x_2 = teacher_att_top_index[i][1]
            student_att_top_3[i] = torch.cat((student_att[i][ x_1+7*y_id[0] : x_2+7*y_id[0] ],
                                              student_att[i][ x_1+7*y_id[1] : x_2+7*y_id[1] ],
                                              student_att[i][ x_1+7*y_id[2] : x_2+7*y_id[2] ],
                                             ))
            teacher_att_top_3[i] = torch.cat((teacher_att[i][ x_1+7*y_id[0] : x_2+7*y_id[0] ],
                                              teacher_att[i][ x_1+7*y_id[1] : x_2+7*y_id[1] ],
                                              teacher_att[i][ x_1+7*y_id[2] : x_2+7*y_id[2] ],
                                             ))
            
        student_att = F.normalize(student_att)
        teacher_att = F.normalize(teacher_att)
        student_att_top_5  = F.normalize(student_att_top_5)
        teacher_att_top_5  = F.normalize(teacher_att_top_5)
        student_att_top_3  = F.normalize(student_att_top_3)
        teacher_att_top_3  = F.normalize(teacher_att_top_3)
            
        # ------------------------------------------------------------------------------------------------
        if   self.att_loss_positive == 1:
            att_loss_per_sample        = (student_att - teacher_att).pow(2).mean(1)
            att_top_5_loss_per_sample  = (student_att_top_5 - teacher_att_top_5).pow(2).mean(1)
            att_top_3_loss_per_sample  = (student_att_top_3 - teacher_att_top_3).pow(2).mean(1)
        elif self.att_loss_positive == 0:
            att_loss_per_sample        = (student_att*teacher_att).sum(1)
            att_top_5_loss_per_sample  = (student_att_top_5*teacher_att_top_5).sum(1)
            att_top_3_loss_per_sample  = (student_att_top_3*teacher_att_top_3).sum(1)
            
        kwargs["_student_logits"] = F.softmax(target_output[0], dim=1)
        kwargs["_teacher_logits"] = F.softmax(source_output[0].detach(), dim=1)
        kwargs["_label_id"] = label_id
            
        att_loss        = self.gate.f(att_loss_per_sample,        log, **kwargs)  # 7*7
        att_top_5_loss  = self.gate.f(att_top_5_loss_per_sample,  log, **kwargs)
        att_top_3_loss  = self.gate.f(att_top_3_loss_per_sample,  log, **kwargs)
          
        # ------------------------------------------------------------------------------------------------
        loss = self.loss_weight * ( (att_loss + att_top_5_loss + att_top_3_loss)/3 )
                
        if log is not None:
            self.ite = self.ite + 1
        
        return loss

    
# KL Distillation Loss + Attention Loss
class KL_AttLoss(_LossBase):
    def __init__(self, args):
        super(KL_AttLoss, self).__init__(args)
        self.T = args.T
        self.loss_weight = args.loss_weight
        self.gate = globals()[args.gate.name](self, args.gate.args)
        self.soft_loss_positive = args.soft_loss_positive
        self.att_loss_positive  = args.att_loss_positive
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        return

    def forward(self, target_output, source_output, label_id, log=None, **kwargs):
        # Distillation Loss ---------------------------------------------------------------------------
        student_logits = target_output[0]
        teacher_logits = source_output[0].detach()
        
        student_softmax = F.softmax(student_logits/self.T, dim=1)
        teacher_softmax = F.softmax(teacher_logits/self.T, dim=1)
        
        if   self.soft_loss_positive == 1:
            soft_loss_per_sample = self.kl_divergence(student_softmax, teacher_softmax, log=log)
        elif self.soft_loss_positive == 0:
            soft_loss_per_sample = self.cos_sim(student_softmax, teacher_softmax)
        
        kwargs["_student_logits"] = student_logits
        kwargs["_teacher_logits"] = teacher_logits
        kwargs["_label_id"] = label_id
        
        soft_loss = self.gate.f(soft_loss_per_sample, log, **kwargs)
                
        kl_loss = soft_loss * (self.T**2)
        
        # Attention loss ------------------------------------------------------------------------------
        student_att, student_att_top_index = target_output[2]
        teacher_att, teacher_att_top_index = source_output[2]
        teacher_att = teacher_att.detach()
        
        # ------------------------------------------------------------------------------------------------
        # 5*5
        student_att_top_5 = torch.zeros(len(student_att), 5*5, device=student_att.device)
        teacher_att_top_5 = torch.zeros(len(teacher_att), 5*5, device=teacher_att.device)
        # 3*3
        student_att_top_3 = torch.zeros(len(student_att), 3*3, device=student_att.device)
        teacher_att_top_3 = torch.zeros(len(teacher_att), 3*3, device=teacher_att.device)
        
        for i in range(len(student_att)):
            # 5*5 -----------------------------------------------------------------------------------
            y_id, x_1, x_2 = teacher_att_top_index[i][0]
            student_att_top_5[i] = torch.cat((student_att[i][ x_1+7*y_id[0] : x_2+7*y_id[0] ],
                                              student_att[i][ x_1+7*y_id[1] : x_2+7*y_id[1] ],
                                              student_att[i][ x_1+7*y_id[2] : x_2+7*y_id[2] ],
                                              student_att[i][ x_1+7*y_id[3] : x_2+7*y_id[3] ],
                                              student_att[i][ x_1+7*y_id[4] : x_2+7*y_id[4] ],
                                             ))
            teacher_att_top_5[i] = torch.cat((teacher_att[i][ x_1+7*y_id[0] : x_2+7*y_id[0] ],
                                              teacher_att[i][ x_1+7*y_id[1] : x_2+7*y_id[1] ],
                                              teacher_att[i][ x_1+7*y_id[2] : x_2+7*y_id[2] ],
                                              teacher_att[i][ x_1+7*y_id[3] : x_2+7*y_id[3] ],
                                              teacher_att[i][ x_1+7*y_id[4] : x_2+7*y_id[4] ],
                                             ))
            # 3*3 -----------------------------------------------------------------------------------
            y_id, x_1, x_2 = teacher_att_top_index[i][1]
            student_att_top_3[i] = torch.cat((student_att[i][ x_1+7*y_id[0] : x_2+7*y_id[0] ],
                                              student_att[i][ x_1+7*y_id[1] : x_2+7*y_id[1] ],
                                              student_att[i][ x_1+7*y_id[2] : x_2+7*y_id[2] ],
                                             ))
            teacher_att_top_3[i] = torch.cat((teacher_att[i][ x_1+7*y_id[0] : x_2+7*y_id[0] ],
                                              teacher_att[i][ x_1+7*y_id[1] : x_2+7*y_id[1] ],
                                              teacher_att[i][ x_1+7*y_id[2] : x_2+7*y_id[2] ],
                                             ))
            
        student_att = F.normalize(student_att)
        teacher_att = F.normalize(teacher_att)
        student_att_top_5  = F.normalize(student_att_top_5)
        teacher_att_top_5  = F.normalize(teacher_att_top_5)
        student_att_top_3  = F.normalize(student_att_top_3)
        teacher_att_top_3  = F.normalize(teacher_att_top_3)
            
        # ------------------------------------------------------------------------------------------------
        if   self.att_loss_positive == 1:
            att_loss_per_sample        = (student_att - teacher_att).pow(2).mean(1)
            att_top_5_loss_per_sample  = (student_att_top_5 - teacher_att_top_5).pow(2).mean(1)
            att_top_3_loss_per_sample  = (student_att_top_3 - teacher_att_top_3).pow(2).mean(1)
        elif self.att_loss_positive == 0:
            att_loss_per_sample        = (student_att*teacher_att).sum(1)
            att_top_5_loss_per_sample  = (student_att_top_5*teacher_att_top_5).sum(1)
            att_top_3_loss_per_sample  = (student_att_top_3*teacher_att_top_3).sum(1)
            
        kwargs["_student_logits"] = F.softmax(target_output[0], dim=1)
        kwargs["_teacher_logits"] = F.softmax(source_output[0].detach(), dim=1)
        kwargs["_label_id"] = label_id
            
        att_loss_all    = self.gate.f(att_loss_per_sample,        log, **kwargs)  # 7*7
        att_top_5_loss  = self.gate.f(att_top_5_loss_per_sample,  log, **kwargs)
        att_top_3_loss  = self.gate.f(att_top_3_loss_per_sample,  log, **kwargs)
          
        att_loss = (att_loss_all + att_top_5_loss + att_top_3_loss)/3
        
        # ------------------------------------------------------------------------------------------------
        loss = soft_loss + (self.loss_weight * att_loss)
                
        if log is not None:
            self.ite = self.ite + 1
        
        return loss
    
    def kl_divergence(self, student, teacher, log=None):
        kl = teacher * torch.log((teacher / (student+1e-10)) + 1e-10)
        kl = kl.sum(dim=1)        
        loss = kl
        return loss


# Gate

# Base
class _BaseGate():
    def __init__(self, parent, args):
        self.parent = parent
        self.args = args
        
    def f(self, loss_per_sample, log, **kwargs):
        return soft_loss


# Through
class ThroughGate(_BaseGate):
    def f(self, loss_per_sample, log, **kwargs):
        soft_loss = loss_per_sample.mean()
        return soft_loss


# Cutoff
class CutoffGate(_BaseGate):
    def f(self, loss_per_sample, log, **kwargs):
        soft_loss = torch.zeros_like(loss_per_sample[0], requires_grad=True).sum()            
        return soft_loss


# Linear
class LinearGate(_BaseGate):
    def f(self, loss_per_sample, log, **kwargs):
        if log is not None:
            self.end_ite = log.iteration
            
        loss_weight = self.parent.ite / self.end_ite
        
        soft_loss = loss_per_sample.mean()
        soft_loss = soft_loss * loss_weight
        
        if log is not None:
            source_id = kwargs["_source_id"]
            target_id = kwargs["_target_id"]
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_linear_weight"] = float(loss_weight)
            
        return soft_loss


# Correct
class CorrectGate(_BaseGate):    
    def f(self, loss_per_sample, log, **kwargs):
        student_logits = kwargs["_student_logits"]
        teacher_logits = kwargs["_teacher_logits"]
        label_id = kwargs["_label_id"]
        
        true_s = student_logits.argmax(dim=1) == label_id
        true_t = teacher_logits.argmax(dim=1) == label_id
        TT = ((true_t == 1) & (true_s == 1)).type_as(loss_per_sample[0])
        TF = ((true_t == 1) & (true_s == 0)).type_as(loss_per_sample[0])
        FT = ((true_t == 0) & (true_s == 1)).type_as(loss_per_sample[0])
        FF = ((true_t == 0) & (true_s == 0)).type_as(loss_per_sample[0])
        mask = 1*TT + 1*TF + 0*FT + 0*FF
        
        soft_loss = (loss_per_sample * mask).mean()
                
        if log is not None:
            source_id = kwargs["_source_id"]
            target_id = kwargs["_target_id"]
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_TT"] = float(TT.sum())
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_TF"] = float(TF.sum())
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_FT"] = float(FT.sum())
            log["ite_log"][self.parent.ite][f"{target_id:02}_{source_id:02}_FF"] = float(FF.sum())
        
        return soft_loss
