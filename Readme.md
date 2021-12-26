# DeepCSeqSite*

#Li Shiwei, Zhong Xianni, Xu Zezhao, Guan Xinglei#

We have improved DeepCSeqSite[(Cui et al., 2019)](https://paperpile.com/c/6re3VV/V7Ig) for our ***Binding site identification with deep learning*** project of SI364. The improvements were dataset, features and loss function. Here are our data, scripts and networks.



# Quick Start

## Requirements

Python = 3.7.x

pytorch = 1.8.x

matplotlib = 3.x

numpy = 1.20.x

scikit-learn = 1.0



## Training model

Our network file is in **./network** folder. All you need to do is run the python file in **./network**. (remember to put data folder in correct location). Our python file will print several indicators to show the performance of model.

If you want to change the dataset, please use scripts in **./scripts/database**(optional) and **./scripts/feature_extraction** to prepare your data.

