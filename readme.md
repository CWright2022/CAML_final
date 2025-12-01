# CSEC 520/620 – Cyber Analytics & Machine Learning  
## Final Project: Data Processing & Application  
### Malicious URL Classification Using Machine Learning

---

## Overview

This project processes raw malicious URL data into a machine-learning–ready format and evaluates multiple ML models—including Decision Trees, Random Forests, and Neural Networks—for detecting and classifying malicious URLs.

The raw dataset undergoes custom cleaning, normalization, and feature extraction implemented in Python, and is evaluated with multiple supervised learning methods.

---

## Project Structure




# Unfinished readme

## Running machine learning models

The neural network (nn.py) in this repository is recommended to be run with CUDA. By default, requirement.txt will install a CPU version of torch (torch=2.9.0).

To enable torch to use your GPU, you must install the appropriate version. For instance, for cu128, use this command:

```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128```

To learn more, go to https://pytorch.org/get-started/locally/