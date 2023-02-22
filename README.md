# Complete and Sound Formal Verification of Adversarial robustness of TTnet with generic SAT solver

## Overview

This repository contains the results of Table 4. For Table 3, see [VNN Github](https://github.com/stanleybak/vnncomp2021)

## Results

The following datasets have been evaluated:

- MNIST
- CIFAR10

### MNIST

On the first 1000 images the performances are:

|                 | Acc. Natural | Acc. Verifiable | Generation time | Solving time | Total Verification Time | Avg. Verification Time / img |
|-----------------|:------------:|:---------------:|:---------------:|:------------:|:-----------------------:|:----------------------------:|
| Low noise 0.1   | 98.0%        |      92.7%      |      10.6s      |     0.4s     |          11.0s          |             11ms             |
| High noise  0.3 |      98.0%        |      55.7%      |      19.6s      |     27.1     |          46.7s          |             47ms             |

### CIFAR10
On the first 1000 images the performances are:

|                   | Acc. Natural | Acc. Verifiable | Generation time | Solving time | Total Verification Time | Avg. Verification Time / img |
|-------------------|:------------:|:---------------:|:---------------:|:------------:|:-----------------------:|:----------------------------:|
| Low noise 2/255   |    53.8%     |      32.3%      |      16.4s      |     2.9s     |          19.3s          |             36ms             |


## Usage

### Configuration
This project uses Python 3.10. To install the required packages, run the following command:

```
pip3 install -r requirements.txt
```

### Running Inference

The pretrained models and truth tables can be downloaded [here](XXX).
This command is very memory comsumming, please make sure that you have at least 50GB of RAM free before starting the experiment.
To run the SAT solver verifier for the first 1K samples.

To run the inference, use the following command:

```
python3 main.py
```

You can parametrize the run as follows

```
--attack_eps_ici 0.1 # noise level [0.1, 0.3] for MNIST and [2,8] for CIFAR10
--solver Minicard # General SAT solver that you can use among ["Minicard", "Glucose3", "Glucose4", "Minisat22", 
                                                                "Lingeling", "CaDiCaL", "MapleChrono", "MapleCM", 
                                                                  "Maplesat", "Mergesat3", "MinisatGH"]
```

### Changing the Dataset

To change the dataset, modify the dataset field in the `config.yaml` file.



