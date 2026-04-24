# Pedestrian Trajectory Prediction with Federated Learning
This repo contains code for our final project for EECE 5643 (Simulation & Performance Evaluation) titled 'Analyzing Federated Learning Performance for Pedestrian Trajectory Prediction Across Motion and Social Factors'. 

## Description

### Datasets
We used pre-annotated versions of the ETH and UCY pedestrian datasets, taken from the implementation in [this paper](https://arxiv.org/pdf/1803.10892) [1]. 

### Model Architecture
We based our model architecture off of that described in [this paper](https://link.springer.com/article/10.1007/s40747-023-01239-5) [2]. It uses an LSTM encoder/decoder to predict the next 12 frames of a pedestrian's trajectory given 8 history frames. 

## Instructions
This simulation is run using [Flower](https://flower.ai/), a Federated Learning simulator. You will need to install it and all required dependencies in order to replicate our results. Once installed (and once this repo is cloned) you should be able to run our simulation by running `flwr run .` at the top level.

To change partitioning schemes, change the value of `sorted_dataset` in `task.py`. Valid dataset files are in the `datasets/sorted_datasets` directory.

## References
[1] A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi, “Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks,” Mar. 29, 2018, arXiv: arXiv:1803.10892. doi: 10.48550/arXiv.1803.10892.   
[2] R. Ni, Y. Lu, B. Yang, C. Yang, and X. Liu, “A federated pedestrian trajectory prediction model with data privacy protection,” Complex Intell. Syst., vol. 10, no. 2, pp. 1787–1799, Apr. 2024, doi: 10.1007/s40747-023-01239-5. 
