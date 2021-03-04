# Coded Machine Unlearning
This repository contains the code used for experiments presented in the paper:

> [**Coded Machine Unlearning**](https://arxiv.org/abs/2012.15721)
> 
> Nasser Aldaghri, Hessam Mahdavifar, and Ahmad Beirami

Machine learning models may store information about individual samples used during the training phase. Some sample may need to be removed from the model for various reasons. Retraining the model after removing such samples from the training dataset ensures complete removal. As more data becomes available, cost of retraining becomes increasingly expensive. Training ensemble models on disjoint shards of the training dataset reduces the cost of unlearning, possibly at the expense of degraded performance ([Bourtoule et al. 2020](https://arxiv.org/abs/1912.03817)).

We propose a utilizing a linear encoder to compress the training dataset into a fewer number of shards used to train the ensemble model for linear and ridge regression. For non-linear regression, random feature projections ([Rahimi and Recht 2008](https://ieeexplore.ieee.org/abstract/document/4797607)) along with ridge regression are utilized.

This repository contains the python code that varies the size of shards and plots the perofrmance vs unlearning cost tradeoff for uncoded machine unlearning ([Bourtoule et al. 2020](https://arxiv.org/abs/1912.03817)) and the proposed coded machine unlearning. As can be seen from experiments, coded unlearning can provide a significant gain in terms of performance vs unlearning cost tradeoff, depending on some propoerties of the features.

# General Guidelines

The code runs the experiment on a either a synthetic dataset or a user-provided dataset. The synthetic data has features drawn from lognormal distribution with parameters mu=1, sigma^2=0.7. The response variable is generated as a polynomial of degree 3 with additive noise.

If user-provided dataset is to be used, it needs to be a numerical matrix without any missing values with rows representing samples and columns representing features and the response variable is the last column in such matrix. The dataset should be named `my_dataset.csv` and located in the same directory as the code.

The goal in this simulation is to observe the behavior of the tradeoff. Using randomly generated synthetic datasets with normalization might change the attainable MSE, but is irrelevent to the our goal which is to observe the behavior of the tradeoff.

The code will output a file named `Output.csv` containing the results of the simulations and two figures: (1) Performance of the testing data vs unlearning cost tradeoff figure named `Test_tradeoff.png`. (2) Performance of training data vs average learning cost figure named `Train_tradeoff.png`.


# Run simulations

To start the simulations, first load dependencies by downloading the file `reqs.txt` and running the following command:
```
pip3 install -r reqs.txt
```
Now, depending on the dataset to be used, either run `synthetic.sh` or `realistic.sh`. The latter requires the desired dataset to be in the same directory and named `my_dataset.csv` and follows the aforementioned guidelines.

**Note:** The code normalizes all features along with the response variable such that each has the range [0,1].

In the shell file you can set the following parameters:
| Parameter  | Description |
| ---------- | ----------- |
| `Alpha`  | Regularization parameter for ridge regression |
| `Code_rate`  | The rate of the encoder; should be greater than 1 |
| `Num_trials`  | Number of simulation runs to generate the curves |
| `Synthetic`  | Set to 1 for lognormal synthetic dataset, 0 for user-provided dataset |
| `Sigma`  | Sigma parameter for lognormal features |
| `Num_features`  | Number of original lognormal features |
| `Num_samples`  | Total number of samples for synthetic dataset |
| `Num_test`  | Number of samples used for testing for any dataset |
| `Random_proj`  | Set to 1 to use random projections corresponding to the Gaussian kernel, 0 to use original features |
| `Show_progress`  | Set to 1 to view the progress and intermediate results while executing the code |

Once the parameters are set, run the code using the following command:
```
bash filename.sh
```

# Sample simulation results:
