# Scene-Text-Classification
Scene Text Classification using Convolutional Recurrent Network

This repostory implements CRNN for Scene Text Classification. The original paper about CRNN can be found [here](http://arxiv.org/abs/1507.05717)

# Software and tools
* Pytorch 1.0
* Python3
* lmdb

# Preparation
Prepare your dataset first, store them in lmdb file.
'''bash
create_dataset.py
'''

# Usage
Simply run train.py to train the model

#TODO
* Evaluate and predict
* Use larger dataset to train the model, like [syn90K](https://www.robots.ox.ac.uk/~vgg/data/text/)
