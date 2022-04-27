# Deep Ensemble Collaborative Learning
Implementation of "Deep Ensemble Collaborative Learning by using Knowledge-transfer Graph for Fine-grained Object Classification" ([arXiv](https://arxiv.org/abs/2103.14845))

## Environment
* python 3.8.12
* ipython 7.27.0
* jupyterlab 2.3.2
* numpy 1.10.0
* sklearn 0.24.2
* pytorch 1.10.0
* torchvision 0.11.0
* optuna 2.10.0
* easydict 1.9
* graphviz 0.17

## Usage
0. Download dataset

1. Optimize graph

    Example of using three models:
~~~ 
ipython optimize_graph.py -- --num_nodes=3 --dataset=StanfordDogs_split --gpu_id=0 --num_trial=6000 --optuna_dir=./optimized_graph/
~~~

2. Confirm the result of optimization

    Open watch_graph.ipynb on jupyterlab and run all cells.

3. Train models by the optimized graph

    Example of using graphs in ’0000’:
~~~ 
ipython train_abn.py -- --num_nodes=3 --dataset=StanfordDogs --gpu_id=0 --target_graph=./optimized_graph/0000/ --save_dir=./result/
~~~
