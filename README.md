# Deep ensemble learning by diverse knowledge distillation
Implementation of "Deep ensemble learning by diverse knowledge distillation for fine-grained object classification"

[ECCV2022 paper] [[arXiv paper](https://arxiv.org/abs/2103.14845)]

## Environment
Our source code is based on [https://github.com/somaminami/DCL](https://github.com/somaminami/DCL) implemented with PyTorch. We are grateful for the author!

Requirements of version are as follows:
- python : 3.8.12
- ipython : 7.27.0
- jupyterlab : 2.3.2
- numpy : 1.10.0
- sklearn : 0.24.2
- pytorch : 1.10.0
- torchvision : 0.11.0
- optuna : 2.10.0
- easydict : 1.9
- graphviz : 0.17

## Dataset
* Stanford Dogs [[link](http://vision.stanford.edu/aditya86/ImageNetDogs/)]
* Caltech-UCSD Birds-200-2011 (CUB-200-2011) [[link](http://www.vision.caltech.edu/datasets/cub_200_2011/)]
* Stanford Cars [[link](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)]
* CIFAR-10
* CIFAR-100

## 1. Optimize graph
Example of using three network nodes.

Optimize on CIFAR dataset:
```bash
ipython optimize_graph_cifar.py -- --num_nodes=3 --dataset=CIFAR10_split --gpu_id=0 --num_trial=6000 --optuna_dir=./optimized_graph/CIFAR10/
```

Optimize on other dataset:
```bash
ipython optimize_graph.py -- --num_nodes=3 --dataset=StanfordDogs_split --gpu_id=0 --num_trial=6000 --optuna_dir=./optimized_graph/SDogs/
```

## 2. Confirm the result of optimization
Open watch_graph.ipynb on jupyterlab and run all cells.

## 3. Train models by the optimized graph
Example of optimized result with "0000" structure.

Train ResNet-18 using Stanford Dogs and graph optimized on Stanford Dogs:
``` bash
ipython train_resnet.py -- --num_nodes=3 --dataset=StanfordDogs --gpu_id=0 --target_graph=./optimized_graph/SDogs/0000/ --save_dir=./result/
```

Train ABN based on ResNet-18 using Stanford Dogs and graph optimized on Stanford Dogs:
``` bash
ipython train_abn.py -- --num_nodes=3 --dataset=StanfordDogs --gpu_id=0 --target_graph=./optimized_graph/SDogs/0000/ --save_dir=./result/
```

Train ABN based on ResNet-18 using CUB-200-2011 and graph optimized on Stanford Dogs:
``` bash
ipython train_abn.py -- --num_nodes=3 --dataset=CUB2011 --gpu_id=0 --target_graph=./optimized_graph/SDogs/0000/ --save_dir=./result/
```

Train ABN based on ResNet-20 using CIFAR-10 and graph optimized on Stanford Dogs:
``` bash
ipython train_cifar_abn.py -- --num_nodes=3 --dataset=CIFAR10 --gpu_id=0 --target_graph=./optimized_graph/SDogs/0000/ --save_dir=./result/
```

## Experiments result
### Optimized graph
| Dataset for optimization | Two nodes | Three nodes | Four nodes | Five node |
|:------------------------:|:---------:|:-----------:|:----------:|:---------:|
| Stanford Dogs | [link](https://www.dropbox.com/sh/xioysci4nqfxpqq/AABUUNGJS7mzjYY2zutkktTIa?dl=1) | [link](https://www.dropbox.com/sh/voht316x3yzl0wu/AABFmUVXOw4kywpy0uPZshaFa?dl=1) | [link](https://www.dropbox.com/sh/z3lelcsjmrq5lys/AABGrjfFpJNeLxhljVTeiRRPa?dl=1) | [link](https://www.dropbox.com/sh/saqcg9jxnfl0p3f/AAAPONdM4OPRQR-NaAFlXq6-a?dl=1) |
| CUB-200-2011  | [link](https://www.dropbox.com/sh/d09r8o5ifua0yua/AACRHT45eAPfrsnYshGv6cJ1a?dl=1) | [link](https://www.dropbox.com/sh/6tx1qfr14mni7rn/AABH3Z_jqawiDMwSqvZMScwwa?dl=1) | [link](https://www.dropbox.com/sh/5zg9nwe2unexn1g/AABhpnOw4VnK5WM8oq1k3oAda?dl=1) | [link](https://www.dropbox.com/sh/9uogmm4s9o4oal0/AADiSHDde2dWbfW5CAbcEcKNa?dl=1) |
| Stanford Cars | [link](https://www.dropbox.com/sh/pst5cpu0dizi1ze/AAD55j_TA0tO_08vvyTIMf2Da?dl=1) | [link](https://www.dropbox.com/sh/tzz08pbntobmmv8/AADWBXFKuyAeGPAXozuBGuUEa?dl=1) | [link](https://www.dropbox.com/sh/r2qvevtu1kk26vl/AAACqWG7utX_nVPLQGIy6MH1a?dl=1) | [link](https://www.dropbox.com/sh/6scxehu6ja9ge6a/AACWjpWj1H_v8jeWhrQm6NKPa?dl=1) |
| CIFAR-10      | [link](https://www.dropbox.com/sh/8pv7g570ewhlurw/AAB8ABN2kHZBD-JaPiiBCvdYa?dl=1) |           - |          - |         - |
| CIFAR-100     | [link](https://www.dropbox.com/sh/2hw4wef1a8ssphc/AADTkVsEwfIT1S3m8ey1W7iua?dl=1) |           - |          - |         - |

### Performances and Trained models

ResNet-18:
|             | No. of<br>nodes | Optimize<br>graph | Train<br>models | config | log | ckpt | Ensemble<br>acc. [%] |
|:-----------:|:-:|:-------------:|:-------------:|:------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|:-----:|
| Independent | 2 |             - | Stanford Dogs | [link](https://www.dropbox.com/s/yk49oq8ujdlrxko/config.json?dl=1) | [link](https://www.dropbox.com/sh/szc6byegepz0llg/AABglo2T_OCcQVlLbkmlP6iva?dl=1) | [link](https://www.dropbox.com/s/z9rmbr6iy3vzl7n/checkpoint_epoch_300.pkl?dl=1) | 68.48 |
| Ours        | 2 | Stanford Dogs | Stanford Dogs | [link](https://www.dropbox.com/s/z25tuc3kgsqtwe7/config.json?dl=1) | [link](https://www.dropbox.com/sh/e73ublg1zwbjqet/AAC9DR9Inf4FTmIgi3xqCJK9a?dl=1) | [link](https://www.dropbox.com/s/i0eakn4lvwji5ps/checkpoint_epoch_300.pkl?dl=1) | 72.60 |
| Independent | 3 |             - | Stanford Dogs | [link](https://www.dropbox.com/s/gufdu5n9tqtr7xb/config.json?dl=1) | [link](https://www.dropbox.com/sh/94m9c2j35ulvmy2/AADTK5dvvEBJEyOUMlcaDu8ha?dl=1) | [link](https://www.dropbox.com/s/9w4fz7liuu4ibtk/checkpoint_epoch_300.pkl?dl=1) | 69.07 |
| Ours        | 3 | Stanford Dogs | Stanford Dogs | [link](https://www.dropbox.com/s/i60qaeorxdt3f78/config.json?dl=1) | [link](https://www.dropbox.com/sh/t7pqt390wt8qip4/AABp-002diSxQyhyAYs_ppW8a?dl=1) | [link](https://www.dropbox.com/s/9uvzgl41s7pby0b/checkpoint_epoch_300.pkl?dl=1) | 72.32 |
| Independent | 4 |             - | Stanford Dogs | [link](https://www.dropbox.com/s/fee1fb6adyytd8i/config.json?dl=1) | [link](https://www.dropbox.com/sh/7yllr74k0mqkpi4/AACOOY7Jcb_pD3eaCFBGGqeIa?dl=1) | [link](https://www.dropbox.com/s/1scxwvfjfesrrp3/checkpoint_epoch_300.pkl?dl=1) | 69.43 |
| Ours        | 4 | Stanford Dogs | Stanford Dogs | [link](https://www.dropbox.com/s/mnkaw8tawwr9aa8/config.json?dl=1) | [link](https://www.dropbox.com/sh/p88y6ztoq1o70ye/AABME5A3eMnj5QiaU_lpDlTWa?dl=1) | [link](https://www.dropbox.com/s/2xf5m0h7sq3vqce/checkpoint_epoch_300.pkl?dl=1) | 72.83 |
| Independent | 5 |             - | Stanford Dogs | [link](https://www.dropbox.com/s/zceecvvj6542vtw/config.json?dl=1) | [link](https://www.dropbox.com/sh/6eu4j2lm95j3xb4/AABlncOgY8gUpAsaYd4gXI9Ta?dl=1) | [link](https://www.dropbox.com/s/ae2pk8c7m1qei96/checkpoint_epoch_300.pkl?dl=1) | 69.60 |
| Ours        | 5 | Stanford Dogs | Stanford Dogs | [link](https://www.dropbox.com/s/imeqvruhua1mh4e/config.json?dl=1) | [link](https://www.dropbox.com/sh/qynz4jroq2b85vg/AADH4xQeTfc5-O6-2Z9nWlvda?dl=1) | [link](https://www.dropbox.com/s/gg1bz2bg5islcn6/checkpoint_epoch_300.pkl?dl=1) | 71.93 |

ABN based on ResNet-18:
|             | No. of<br>nodes | Optimize<br>graph | Train<br>models | config | log | ckpt | Ensemble<br>acc. [%] |
|:-----------:|:-:|:-------------:|:-------------:|:------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|:-----:|
| Independent | 2 |             - | Stanford Dogs | [link](https://www.dropbox.com/s/ce6yjfj6np2yo15/config.json?dl=1) | [link](https://www.dropbox.com/sh/jdt1pxpb9t7smce/AABze-fPZ00w5-bFl1o6Q06Ua?dl=1) | [link](https://www.dropbox.com/s/9kh9zgkixcnfc6w/checkpoint_epoch_300.pkl?dl=1) | 71.06 |
| Ours        | 2 | Stanford Dogs | Stanford Dogs | [link](https://www.dropbox.com/s/wjn0yutkrtye11j/config.json?dl=1) | [link](https://www.dropbox.com/sh/zux8hc9seq603nb/AAB1LdikoJ0N2W0McDiqYlYEa?dl=1) | [link](https://www.dropbox.com/s/b20t73jqukm2quy/checkpoint_epoch_300.pkl?dl=1) | 74.20 |
| Independent | 3 |             - | Stanford Dogs | [link](https://www.dropbox.com/s/2flgrqba0o56du2/config.json?dl=1) | [link](https://www.dropbox.com/sh/eoi81dnla1tylst/AACrrJGoebqb_SMdH6wyqcX4a?dl=1) | [link](https://www.dropbox.com/s/cr1kufsvuxesm46/checkpoint_epoch_300.pkl?dl=1) | 71.85 |
| Ours        | 3 | Stanford Dogs | Stanford Dogs | [link](https://www.dropbox.com/s/cf0qkghkjtlj2se/config.json?dl=1) | [link](https://www.dropbox.com/sh/u6yha9tfxjb92yq/AAANjW4sVHuiwXik5oGK2uMta?dl=1) | [link](https://www.dropbox.com/s/fhmz4exm7bc1xhg/checkpoint_epoch_300.pkl?dl=1) | 74.01 |
| Independent | 4 |             - | Stanford Dogs | [link](https://www.dropbox.com/s/6l4kaowj1egwmuh/config.json?dl=1) | [link](https://www.dropbox.com/sh/f2gb9u9searxzmu/AACHkk3xcR5iot4kDs-aU1Yda?dl=1) | [link](https://www.dropbox.com/s/sbyyos1qltv005g/checkpoint_epoch_300.pkl?dl=1) | 72.69 |
| Ours        | 4 | Stanford Dogs | Stanford Dogs | [link](https://www.dropbox.com/s/dkqyinpkb3stpd5/config.json?dl=1) | [link](https://www.dropbox.com/sh/t4hwgadlhi560h5/AAAokxWVFEv4QrNR-CR8y-nza?dl=1) | [link](https://www.dropbox.com/s/4fkfvht7tphiwkj/checkpoint_epoch_300.pkl?dl=1) | 74.23 |
| Independent | 5 |             - | Stanford Dogs | [link](https://www.dropbox.com/s/qbm98ku0qvmvohh/config.json?dl=1) | [link](https://www.dropbox.com/sh/extwlhii4ua6qfd/AAC5B4BizrYXYZFFjoN6hU70a?dl=1) | [link](https://www.dropbox.com/s/m86k3s767r88l8u/checkpoint_epoch_300.pkl?dl=1) | 72.61 |
| Ours        | 5 | Stanford Dogs | Stanford Dogs | [link](https://www.dropbox.com/s/3ymozw0td7wvmc2/config.json?dl=1) | [link](https://www.dropbox.com/sh/96yjaw2s5ha3jaz/AABpW8GouU3gB1fVKuT0mWOJa?dl=1) | [link](https://www.dropbox.com/s/n3qti2cs5syp7gv/checkpoint_epoch_300.pkl?dl=1) | 74.52 |

If you try the trained models using the other datasets, please see [[coming soon](https://github.com/naok0615/ensemble_graph/blob/main/TRAINED.md)].
