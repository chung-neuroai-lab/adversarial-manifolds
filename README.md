# Adversarial Manifolds
This repository accompanies the paper ["Neural Population Geometry Reveals the Role of Stochasticity in Robust Perception"](https://openreview.net/forum?id=BfcE_TDjaG6) providing example code to demonstrate our analysis.

MFTMA_analyze_adversarial_representations.ipynb provides examples of
 - loading CIFAR10 VOneResNet18 and regular ResNet18 (adversarially trained ResNet18 coming)
 - constructing class or exemplar manifolds
 - using ART to attack the resultant networks
 - feature extraction and preprocessing for analysis
 - using MFTMA or SVM analysis to analyze class or exemplar manifolds representations
 - plotting the results

Stay tuned for more notebooks, showing how to generate VOneResNet18 with Gaussian noise for CIFAR10, and other demos.

## Installation

Code is tested using Python 3.7.11. 

If using conda, create a conda environment and activate with: 
```
conda create -n adv_manifolds python=3.7
conda activate adv_manifolds
```

Install the requirements and package using: 
```
pip install -e .
```

## Citation
If you find this code useful for your research, please cite our paper: 
```
@inproceedings{
dapello2021neural,
title={Neural Population Geometry Reveals the Role of Stochasticity in Robust Perception},
author={Joel Dapello and Jenelle Feather and Hang Le and Tiago Marques and David Daniel Cox and Josh Mcdermott and James J. DiCarlo and SueYeon Chung},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=BfcE_TDjaG6}
}
```
