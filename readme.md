# Unfinished readme

## Running machine learning models

The neural network (nn.py) in this repository is recommended to be run with CUDA. By default, requirement.txt will install a CPU version of torch (torch=2.9.0).

To enable torch to use your GPU, you must install the appropriate version. For instance, for cu128, use this command:

```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128```

To learn more, go to https://pytorch.org/get-started/locally/