# coding: utf-8

import os
import random
import easydict
import copy
import argparse
import torch
import torch.nn as nn
import optuna

from lib import dataset_factory
from lib import models as model_fuctory
from lib import loss_func_cifar_abn as loss_func
from lib import trainer_cifar_abn as trainer_module
from lib import utils


parser = argparse.ArgumentParser()

parser.add_argument('--num_nodes', type=int, default=3)
parser.add_argument('--dataset', type=str, choices=["CIFAR10_split","CIFAR100_split"], default="CIFAR10_split")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--num_trial', type=int, default=6000)
parser.add_argument('--optuna_dir', type=str, default="./optimized_graph/001/")

try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args(args=[
        "--num_nodes", "2",
        "--dataset", "CIFAR10_split",
        "--gpu_id", "0",
        "--num_trial", "6000",
        "--optuna_dir", "./optimized_graph/001/",
    ])

get_ipython().magic('env CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().magic('env CUDA_VISIBLE_DEVICES=$args.gpu_id')


# Set config
manualSeed = 0
 
if args.dataset == "CIFAR10_split":
    DATA_PATH  = './dataset/CIFAR10/'
    NUM_CLASS  = 10
    SCHEDULE   = [150,225]
    EPOCHS     = 300
    BATCH_SIZE = 128
elif args.dataset == "CIFAR100_split":
    DATA_PATH  = './dataset/CIFAR100/'
    NUM_CLASS  = 100
    SCHEDULE   = [150,225]
    EPOCHS     = 300
    BATCH_SIZE = 128

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
            "saving_interval": 1000,
            "base_dir": "./",
        },
        #--------------------------Models & Optimizer-------------------------
        "models": 
        [           
            #----------------Model------------------
            {
                "name": "model",
                "args":
                {
                    "num_classes": NUM_CLASS,
                },
                "load_weight":
                {
                    "path": None,
                    "model_id": 0,
                },
                "optim": optim_setting,
            } 
            for _ in range(args.num_nodes+1)
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
                None
                for _ in range(args.num_nodes+1)
            ]
            for _ in range(args.num_nodes+1)
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
    #torch.manual_seed(config.manualSeed)
    #torch.cuda.manual_seed_all(config.manualSeed)
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


# Optuna

# Args Factory
args_factory = easydict.EasyDict({
    "models": {
        "ResNet20_ABN":
        {
            "name": "resnet20_abn",
            "args":
            {
                "num_classes": NUM_CLASS,
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


# Inform function for optuna
def inform_optuna(**kwargs):
    trial = kwargs["_trial"]
    logs  = kwargs["_logs"]
    epoch = kwargs["_epoch"]
    
    error = 100 - logs[0]["epoch_log"][epoch]["test_accuracy"]
    trial.report(error, step=epoch)
    
    if trial.should_prune():
        raise optuna.structs.TrialPruned()
    return


# Hyperparameters
LOSS_LISTS = [
    [
        ["IndepLoss"] if i == j else ["KLLoss_P", "KLLoss_N", "AttLoss_P", "AttLoss_N", "KL_Att_P_P", "KL_Att_N_N"]
        for i in range(args.num_nodes+1)
    ]
    for j in range(args.num_nodes+1)
]

for i in range(args.num_nodes+1):
    if i == 0:
        for j in range(args.num_nodes):
            LOSS_LISTS[i][j+1] = ["KLLoss_P"]
    else:
        LOSS_LISTS[i][0] = ["KLLoss_P"]

GATE_LIST = [
    [
        ["ThroughGate", "CutoffGate", "CorrectGate", "LinearGate"]  if not i == 0 else ["CutoffGate"]
        for i in range(args.num_nodes+1)
    ]
    for j in range(args.num_nodes+1)
]


GATE_LIST[0] = [
    ["CutoffGate"]
    for i in range(args.num_nodes+1)
]

MODEL_LISTS = [
    ["Ensemble"]
]+[
    ["ResNet20_ABN"]
    for i in range(args.num_nodes)
]


# Objective function for optuna
def objective_func(trial):
    global config
        
    if type(args.num_trial) is int:
        if trial.number >= args.num_trial:
            import sys
            sys.exit()
        
    # make dirs
    config.trainer.base_dir = os.path.join(args.optuna_dir, f"{trial.number:04}/")
    utils.make_dirs(config.trainer.base_dir+"log")
    utils.make_dirs(config.trainer.base_dir+"checkpoint")
    
    # change config        
    # set loss funcs & gates
    for source_id, model_losses in enumerate(config.losses):
        for target_id, _ in enumerate(model_losses):
            loss_name = trial.suggest_categorical(f'{source_id:02}_{target_id:02}_loss',
                                                  LOSS_LISTS[source_id][target_id])
            
            loss_args = copy.deepcopy(args_factory.losses[loss_name])
            if "gate" in loss_args.args:
                gate_name = trial.suggest_categorical(f'{source_id:02}_{target_id:02}_gate', GATE_LIST[source_id][target_id])
                loss_args.args.gate = copy.deepcopy(args_factory.gates[gate_name])
            config.losses[source_id][target_id] = loss_args
    
    for model_id in range(len(config.models)):
        # set model
        model_name = trial.suggest_categorical(f"model_{model_id}_name", MODEL_LISTS[model_id])
        model = copy.deepcopy(args_factory.models[model_name])
        config.models[model_id].name = model.name
        config.models[model_id].args = model.args
        
        # set model weight
        is_ensemble = config.models[model_id].name == "Ensemble"
        if (not is_ensemble):
            config.models[model_id].load_weight.path = model.load_weight.path
    
    config = copy.deepcopy(config)
    
    # save config
    utils.save_json(config, config.trainer.base_dir+r"log/config.json")
    
    # create object
    trainer, nets, criterions, optimizers, train_loader, test_loader, logs = create_object(config)

    # make kwargs
    kwargs = {"_trial": trial,
              "_callback":inform_optuna,
              }
    
    # set seed
    trial.set_user_attr("seed", config.manualSeed)
    
    # raise exception if target model is pretrained.
    if config.models[0].load_weight.path is not None:
        class BlacklistError(optuna.structs.OptunaError):
            pass
        raise BlacklistError()
    
    # start trial
    trainer.train(nets, criterions, optimizers, train_loader, test_loader, logs, trial=trial, **kwargs)
    
    # error of ensemble
    acc = 100 - logs[0]["epoch_log"][config.trainer.epochs]["test_accuracy"]
    
    return acc


# Cteate study object
utils.make_dirs(args.optuna_dir)

sampler = optuna.samplers.RandomSampler()
pruner  = optuna.pruners.SuccessiveHalvingPruner(min_resource=1,
                                                 reduction_factor=2,
                                                 min_early_stopping_rate=0)

db_path = os.path.join(args.optuna_dir, "optuna.db")
study   = optuna.create_study(storage=f"sqlite:///{db_path}",
                              study_name='experiment01',
                              sampler=sampler,
                              pruner=pruner,
                              direction="minimize",
                              load_if_exists=True)

# Start optimization
study.optimize(objective_func, n_trials=None, timeout=None, n_jobs=1)
