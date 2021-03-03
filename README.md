# Coded Machine Unlearning
This repository contains the code used for experiments presented in the paper:

> [**Coded Machine Unlearning**](https://arxiv.org/abs/2012.15721)
> 
> Nasser Aldaghri, Hessam Mahdavifar, and Ahmad Beirami

Machine learning models may store information about individual samples used during the training phase. Some sample may need to be removed from the model for various reasons. Retraining the model after removing such samples from the training dataset ensures complete removal. As more data becomes available, cost of retraining becomes increasingly expensive. Training ensemble models on disjoint shards of the training dataset reduces the cost of unlearning, possibly at the expense of degraded performance ([Bourtoule et al. 2020](https://arxiv.org/abs/1912.03817)).

We propose a utilizing a linear encoder to compress the training dataset into a fewer number of shards used to train the ensemble model for linear and ridge regression. For non-linear regression, random feature projections ([Rahimi and Recht 2008](https://ieeexplore.ieee.org/abstract/document/4797607)) along with ridge regression are utilized.

This repository contains the python code that varies the size of shards and plots the perofrmance vs unlearning cost tradeoff for uncoded machine unlearning ([Bourtoule et al. 2020](https://arxiv.org/abs/1912.03817)) and the proposed coded machine unlearning. As can be seen from experiments, coded unlearning can provide a significant gain in terms of performance vs unlearning cost tradeoff, depending on some propoerties of the features.

# General Guidelines

The code runs the experiment on a synthetic dataset with features drawn from lognormal distribution with parameters mu=1, sigma^2=0.7. The response variable is generated as a polynomial of degree 3 with additive noise.

The goal in this simulation is to observe the behavior of the tradeoff. Using randomly generated synthetic datasets with normalization might change the attainable MSE, but is irrelevent to the our goal which is to observe the behavior of the tradeoff.


# Run The Code on A Synthetic Dataset

```
pip3 install -r reqs.txt
```
