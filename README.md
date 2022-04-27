# Deep Ensemble Collaborative Learning
Implementation of "Deep Ensemble Collaborative Learning by using Knowledge-transfer Graph for Fine-grained Object Classification" ([arXiv](https://arxiv.org/abs/2103.14845))

## Environment
* python
* ipython
* jupyterlab
* pytorch
* torchvision
* optuna
* easydict
* graphviz

## Usage
0. Download dataset

1. Optimize graph
~~~ 
ipython optimize_graph.py -- --num_nodes=3 --dataset=StanfordDogs_split --gpu_id=0 --num_trial=6000 --optuna_dir ./optimized_graph
~~~

2. Confirm the result of optimization

Open watch.ipynb on jupyterlab and run all cells.

3. Train models by the optimized graph
~~~ 
ipython
~~~
