# Residual Attention-based LSTM for Video Caption

An implementation for paper "Residual Attention-based LSTM for Video Caption": https://link.springer.com/article/10.1007%2Fs11280-018-0531-z

## Requirements

 Python 2.7.6

 Theano 0.8.2

## processed data

 You need to download pretrained resnet model for extracting features.
 We provide our extracted ResNet video feature and processed caption in:https://drive.google.com/open?id=1HymvVvAEygM6UJm41dQkQ4IbTWcHT0iQ. Download this dataset and replace RAB_FEATURE_BASE_PATH in config.py with your feature path and replace RAB_DATASET_BASE_PATH in config.py with your processed data path. Besides, you should assign where to store your result in config.py.

## Evaluation

 If you'd like to evaluate BLEU/METEOR/CIDER scores during training. Don't forget
 to download coco-caption:https://github.com/tylin/coco-caption and Jobman:http://deeplearning.net/software/jobman/install.html.
 Also you should add coco-caption path to $PYTHONPATH and add jobman path to $PYTHONPATH as well.

## Others

 If you have any questions, drop us email at:xiangpengli.cs@gmail.com
