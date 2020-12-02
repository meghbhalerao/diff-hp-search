# Differentiable search of meta-parameters in neural networks
The experiments in both the cases below are run using the **CIFAR-10** dataset (the 50,000 training examples split differently according to the experiments).
## Brief Description
This is the code for a small coding experiment that I did, which does a **differentiable hyperparameter search**, to find the weights for each sample in the training dataset, using the validation loss as the tuning parameter. Some part of the code is taken from [1] and [2]. This experiment has 2 parts, one for finding the weights using fully **supervised** training and other using **semi-supervised** methods. 
## Steps to run the code
1. Download **CIFAR-10** or any other dataset according to your liking and split them into train and val (for `main_hp.py`), or train, train_unlabelled, val (for `python main_hp_two_stage.py`)
2. To run the code to find the weights for each example: `python main_hp.py`
3. To run the search using the tri-level optimization: `python main_hp_two_stage.py`
## References
1. Momentum Contrast for Unsupervised Visual Representation Learning. _Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick. CVPR 2020._ 
2. DARTS: Differentiable Architecture Seach. _Hanxiao Liu, Karen Simonyan, Yiming Yang. ICLR 2019._ 
