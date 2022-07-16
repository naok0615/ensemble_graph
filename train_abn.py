# coding: utf-8

import os
import random
import easydict
import copy
import argparse
import torch
import torch.nn as nn
import optuna
import json
from easydict import EasyDict as edict

from lib import dataset_factory
from lib import models as model_fuctory
from lib import loss_func_abn as loss_func
from lib import trainer_abn as trainer_module
from lib import utils


parser = argparse.ArgumentParser()

parser.add_argument('--num_nodes', type=int, default=2)
parser.add_argument('--target_graph', type=str, default="./optimized_graph/SDogs/2models/0000/")
parser.add_argument('--dataset', type=str, choices=["StanfordDogs","StanfordCars","CUB2011"], default="StanfordDogs")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="./result/SDogs_graph/2models/1/")

try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args(args=[
        "--num_nodes", "2",
        "--target_graph", "./optimized_graph/SDogs/2models/0000/",
        "--dataset", "StanfordDogs",
        "--gpu_id", "0",
        "--save_dir", "./result/SDogs_graph/2models/1/",
    ])

get_ipython().magic('env CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().magic('env CUDA_VISIBLE_DEVICES=$args.gpu_id')


# Set config
manualSeed = 0

print(args.dataset)
if args.dataset == "StanfordDogs":
    DATA_PATH  = './dataset/StanfordDogs/'
    NUM_CLASS  = 120
    SCHEDULE   = [150,225]
    EPOCHS     = 300
    BATCH_SIZE = 16
    ckpt_path  = "checkpoint/checkpoint_epoch_300.pkl"
elif args.dataset == "StanfordCars":
    DATA_PATH  = './dataset/StanfordCars/'
    NUM_CLASS  = 196
    SCHEDULE   = [150,225]
    EPOCHS     = 300
    BATCH_SIZE = 16
    ckpt_path  = "checkpoint/checkpoint_epoch_300.pkl"
elif args.dataset == "CUB2011":
    DATA_PATH  = './dataset/cub2011/'
    NUM_CLASS  = 200
    SCHEDULE   = [150,225]
    EPOCHS     = 300
    BATCH_SIZE = 16
    ckpt_path  = "checkpoint/checkpoint_epoch_300.pkl"
    
optim_setting = {
    "name": "SGD",
    "args":
    {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "nesterov": False,
    },
    "schedule": SCHEDULE,
    "gammas": [0.1,0.1],
}


# Args Factory
args_factory = easydict.EasyDict({
    "models": {
        "ResNet18_ABN":
        {
            "name": "resnet18_abn",
            "args":
            {
                "num_classes": NUM_CLASS,
                "pre_ABN":False,
                "pre_ResNet":False,
            },
            "load_weight":
            {
                "path": None,
            },
        },
        "Ensemble":
        {
            "name": "Ensemble",
            "args":
            {
                "source_list": list(range(1, args.num_nodes+1)),
                "detach_list": list(range(1, args.num_nodes+1)),
            },
        },
    },
    "losses":
    {
        "IndepLoss":
        {
            "name": "IndependentLoss",
            "args": 
            {
                "loss_weight": 1,
                "gate": {},
            },
        },
        "KLLoss_P":
        {
            "name": "KLLoss",
            "args":
            {
                "T": 1,
                "loss_weight": 1,
                "gate": {},
                "soft_loss_positive":1,
            },        
        },
        "KLLoss_N":
        {
            "name": "KLLoss",
            "args":
            {
                "T": 1,
                "loss_weight": 1,
                "gate": {},
                "soft_loss_positive":0,
            },        
        },
        "AttLoss_P":
        {
            "name": "AttLoss",
            "args":
            {
                "loss_weight": 1,
                "gate": {},
                "att_loss_positive":1,
            },        
        },
        "AttLoss_N":
        {
            "name": "AttLoss",
            "args":
            {
                "loss_weight": 1,
                "gate": {},
                "att_loss_positive":0,
            },        
        },
        "KL_Att_P_P":
        {
            "name": "KL_AttLoss",
            "args":
            {
                "T": 1,
                "loss_weight": 1,
                "gate":{},
                "soft_loss_positive":1,
                "att_loss_positive" :1,
            },
        },
        "KL_Att_N_N":
        {
            "name": "KL_AttLoss",
            "args":
            {
                "T": 1,
                "loss_weight": 1,
                "gate":{},
                "soft_loss_positive":0,
                "att_loss_positive" :0,
            },
        },
    },
    "gates":
    {
        "CutoffGate":
        {
            "name": "CutoffGate",
            "args": {},
        }, 
        "ThroughGate":
        {
            "name": "ThroughGate",
            "args": {},
        }, 
        "CorrectGate": 
        {
            "name": "CorrectGate",
            "args": {},    
        },
        "LinearGate":
        {
            "name": "LinearGate",
            "args": {},
        },
    },
})


# Load Config
load_config_path = args.target_graph+'log/config.json'
json_open = open(load_config_path, 'r')
json_load = json.load(json_open)

opt_config = edict(json_load)

for i in range(len(opt_config.models)):
    if opt_config.models[i].name == 'resnet20_abn':
        opt_config.models[i].name = 'resnet18_abn'
        opt_config.models[i].args = args_factory.models.ResNet20_ABN.args
    elif 'num_classes' in opt_config.models[i].args:
        opt_config.models[i].args.num_classes = NUM_CLASS


config = easydict.EasyDict(
    {
        #------------------------------Others--------------------------------        
        "doc": "",
        "manualSeed": manualSeed,
        #------------------------------Dataloader--------------------------------
        "dataloader": 
        {
            "name": args.dataset,
            "data_path": DATA_PATH,
            "num_class": NUM_CLASS,
            "batch_size": BATCH_SIZE,
            "workers": 10,
            "train_shuffle": True,
            "train_drop_last": True, 
            "test_shuffle": True,
            "test_drop_last": False, 
        },
        #------------------------------Trainer--------------------------------
        "trainer": 
        {
            "name": "ClassificationTrainer",
            "start_epoch": 1,
            "epochs": EPOCHS,
            "saving_interval": EPOCHS,
            "base_dir": "./",
        },
        #--------------------------Models & Optimizer-------------------------
        "models": 
        [           
            #----------------Model------------------
            {
                "name": opt_config.models[i].name,
                "args": opt_config.models[i].args,
                "load_weight":
                {
                    "path": None,
                    "model_id": 0,
                },
                "optim": optim_setting,
            } 
            for i in range(args.num_nodes+1) # number of network nodes + ensemble node
        ],
        #-----------------------------Loss_func-------------------------------
        #
        #    source node -> target node
        #    [
        #        [1->1, 2->1, 3->1],
        #        [1->2, 2->2, 3->2],
        #        [1->3, 2->3, 3->3],
        #    ]
        #
        "losses":        
        [
            [
                edict(opt_config.losses[k][j])
                for j in range(args.num_nodes+1)
            ]
            for k in range(args.num_nodes+1)
        ],
        #------------------------------GPU-------------------------------- 
        "gpu":
        {
            "use_cuda": True,
            "ngpu": 1,
            "id": 0,
        },
    })

config = copy.deepcopy(config)


# Create object
def create_object(config):
    # set seed value
    config.manualSeed = random.randint(1,10000)
    random.seed(config.manualSeed)
    torch.cuda.manual_seed_all(config.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # load dataset
    train_loader, test_loader = getattr(dataset_factory, config.dataloader.name)(config)    
    
    # model & loss func & optimizer    
    nets = []
    criterions = []
    optimizers = []
    for model_args in config.models:
        # model
        net = getattr(model_fuctory, model_args.name)(**model_args.args)
        
        # load weight
        if model_args.load_weight.path is not None:
            utils.load_model(net, model_args.load_weight.path, model_args.load_weight.model_id)
                    
        net = net.cuda(config.gpu.id)
        
        nets += [net]
        
        # loss function        
        criterions = []
        for row in config.losses:
            r = []
            for loss in row:
                criterion = getattr(loss_func, loss.name)(loss.args)
                criterion = criterion.cuda(config.gpu.id)
                r += [criterion]
            criterions += [loss_func.TotalLoss(r)]
        
        # optimizer
        optimizer = getattr(torch.optim, model_args.optim.name)
        optimizers += [optimizer(net.parameters(), **model_args.optim.args)]
    
    # trainer
    trainer = getattr(trainer_module, config.trainer.name)(config)

    # logger
    logs = utils.LogManagers(len(config.models), len(train_loader.dataset),
                                config.trainer.epochs, config.dataloader.batch_size)

    return trainer, nets, criterions, optimizers, train_loader, test_loader, logs


# Create Save Directory
config.trainer.base_dir = args.save_dir
utils.make_dirs(config.trainer.base_dir+"log")
utils.make_dirs(config.trainer.base_dir+"checkpoint")

utils.save_json(config, config.trainer.base_dir+r"log/config.json")

trainer, nets, criterions, optimizers, train_loader, test_loader, logs = create_object(config)


# Start Study
trainer.train(nets, criterions, optimizers, train_loader, test_loader, logs)
