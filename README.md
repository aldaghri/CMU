# Coded Machine Unlearning
This repository contains the code used for experiments presented in the paper:

> [**Coded Machine Unlearning**](https://arxiv.org/abs/2012.15721)
> 
> Nasser Aldaghri, Hessam Mahdavifar, and Ahmad Beirami

Machine learning models may store information about individual samples used during the training phase. Some samples may need to be removed from the model for various reasons. Retraining the model after removing such samples from the training dataset ensures complete removal. As more data becomes available, the cost of retraining becomes increasingly expensive. Training ensemble models on disjoint shards of the training dataset reduces the cost of unlearning, possibly at the expense of degraded performance ([Bourtoule et al. 2020](https://arxiv.org/abs/1912.03817)).

We propose utilizing a linear encoder to compress the training dataset into a fewer number of shards used to train the ensemble model for linear and ridge regression. For non-linear regression, random feature projections ([Rahimi and Recht 2008](https://ieeexplore.ieee.org/abstract/document/4797607)) along with ridge regression are utilized.

This repository contains the python code that sweeps the size of shards and plots the performance vs. unlearning cost tradeoff for uncoded machine unlearning ([Bourtoule et al. 2020](https://arxiv.org/abs/1912.03817)) and the proposed coded machine unlearning. As can be seen from experiments, coded unlearning can provide a significant gain in terms of performance vs unlearning cost tradeoff, depending on some properties of the features.

# General Guidelines

The code runs the experiment on either a synthetic dataset or a user-provided dataset. The synthetic data has features drawn from lognormal distribution with parameters mu=1, sigma=0.5. The response variable is generated as a polynomial of degree 3 whose coefficients are generated from a normal distribution along with additive noise.

If a user-provided dataset is to be used, it needs to be a numerical matrix without any missing values with rows representing samples and columns representing features and the response variable is the last column in such matrix. The dataset should be named `my_dataset.csv` and located in the same directory as the code.

The goal of this simulation is to observe the behavior of the tradeoff. Using synthetic datasets randomly generated each time the code is run along with normalization might change the value of the attainable MSE. However, this is irrelevant to the goal here as we are concerned with examining the behavior of the tradeoff compared to the unsharded performance (The rightmost point in the uncoded curves).

The code will output a file named `Output.csv` containing the results of the simulations and two figures: (1) Performance of the testing data vs. unlearning cost tradeoff figure named `Test_tradeoff.png`. (2) Performance of training data vs. average learning cost figure named `Train_tradeoff.png`.


# Run simulations

To start the simulations, first load dependencies by downloading the file `reqs.txt` and running the following command:
```
pip3 install -r reqs.txt
```
Now, depending on the dataset to be used, either run `synthetic.sh` or `realistic.sh`. The latter requires the desired dataset to be in the same directory and named `my_dataset.csv` and follows the aforementioned guidelines.

**Note:** The code normalizes all features along with the response variable such that each has the range [0,1].

In the shell files you can specify the following parameters:
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
Running the shell files `synthetic.sh` and `realistic.sh` will use the following parameters, respectively.
<table>
<tr><th>synthetic.sh</th><th>realistic.sh</th></tr>
<tr><td>
  
| Parameter  | Value |
| ---------- | ----------- |
| `Alpha`  | 0 |
| `Code_rate`  | 5 |
| `Num_trials`  | 2000 |
| `Synthetic`  | 1 |
| `Sigma`  | 0.5 |
| `Num_features`  | 50 |
| `Num_samples`  | 15000 |
| `Num_test`  | 2000 |
| `Random_proj`  | 0 |
| `Show_progress`  | 0 |

</td><td>

| Parameter  | Value |
| ---------- | ----------- |
| `Alpha`  | 0.001 |
| `Code_rate`  | 5 |
| `Num_trials`  | 5000 |
| `Synthetic`  | 0 |
| `Sigma`  | -- |
| `Num_features`  | -- |
| `Num_samples`  | -- |
| `Num_test`  | 692 |
| `Random_proj`  | 0 |
| `Show_progress`  | 0 |

</td></tr> </table>

**Note:** Running realistic.sh requires downloading `/Datasets/compacts_projected.csv` then renaming it to `my_dataset.csv`, and commenting lines 231-232 in `CMU.py`.

The results of these simulations are as follows

<table>
<tr><th>Synthetic</th><th>CompAct</th></tr>
<tr><td>

<img src="https://user-images.githubusercontent.com/79866053/109908552-7ec6b600-7c72-11eb-996d-4ce2ad9c1b89.png">

<img src="https://user-images.githubusercontent.com/79866053/109908579-89814b00-7c72-11eb-869c-dbb6c6f6b2e9.png">

</td><td>
  
<img src="https://user-images.githubusercontent.com/79866053/109910167-8045ad80-7c75-11eb-87e1-4add450cea1f.png">

<img src="https://user-images.githubusercontent.com/79866053/109910168-80de4400-7c75-11eb-8d03-925d795f0725.png">

</td></tr> </table>

**For additional discussion and experiments, please refer to the [paper](https://arxiv.org/abs/2012.15721).**
