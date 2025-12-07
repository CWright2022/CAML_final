# CSEC 520/620 – Cyber Analytics & Machine Learning  
## Final Project: Data Processing & Application  
### Malicious URL Classification Using Machine Learning

---

## Overview

This project processes raw malicious URL data into a machine-learning–ready format and evaluates multiple ML models, including decision trees, random forests, and neural networks, for detecting and classifying malicious URLs.

The raw dataset undergoes custom cleaning, normalization, and feature extraction implemented in Python, and is evaluated with multiple supervised learning methods.

---

## Project Structure

- ```data_statistics.py```: Performs statistics on the URL dataset. Holds logic for extracting many of the features and generates charts showing the distribution of those features across the dataset.

- ```decision_tree.py```: Defines, trains and evaluates the decision tree model.

- ```feature_extraction.py```: Extract our features from the dataset and put into X and y structures for machine learning.

- ```load_data.py```: Loads data. Downloads the dataset from Kaggle if needed. Also contains functionality for cleaning and balancing the dataset.

- ```nn.py```: Defines, trains, and evaluates the neural network.

- ```random_forest.py```: Defines, trains, and evaluates the random forest model.

- ```main.py```: Main driver.

## How to Run

Our development was done with Python 3.13.3, although most reasonably recent versions of Python 3 should work fine. Install the necessary requirements from ```requirements.txt```. For example:

```python3 -m pip install -r requirements.txt```

Run ```python3 main.py --help``` to view options. Choose one of the following switches in order to run your desired functionality.

```
usage: Argument parser to help with only running certain parts of the code [-h] [--statistics] [--decision_tree] [--random_forest] [--nn_binary] [--nn_malicious] [--nn_all]

options:
  -h, --help           show this help message and exit
  --statistics, -s     Loads data and does statistics
  --decision_tree, -d  Loads data, extracts features, and trains a decision tree
  --random_forest, -r  Loads data, extracts features, and trains a Random Forest
  --nn_binary, -n      Loads data, extracts features, and trains a neural network to distinguish benign and malicious URLs
  --nn_malicious, -m   Loads data, extracts features, and trains a neural network to distinguish between malicious URLs
  --nn_all, -a         Loads data, extracts features, and trains a neural network to distinguish between all URL types
```


### Note

For efficiency and speed, it is recommended to run the neural network utilizing a GPU with CUDA. However, by default, requirements.txt will install a CPU-only version of torch (torch=2.9.0).

To enable torch to use your GPU, you must install the appropriate version. For instance, for cu128, use this command:

```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128```

To learn more, go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)