#!/usr/bin/env /home/priya/softwares/envs/pytorch_env/bin/python

from sys import argv
import numpy as np
import random
from numpy import load
from pandas import read_csv
import matplotlib.pyplot as plt

##loading data and label files
data1 = load('npz-1.npz')
##checking npz file contents
lst1 = data1.files
a1 = data1['arr_0']
#checking the data structure 
a1.shape

lab1 = read_csv("../csv/csv-1.csv")
#extracting the lable column
lab1 = lab1.values[:, 1]


data2 = load('npz-2.npz')
lst2 = data2.files
a2 = data2['arr_0']
a2.shape

lab2 = read_csv("../csv/csv-2.csv")
lab2 = lab2.values[:, 1]

data3 = load('npz-3.npz')
lst3 = data3.files
a3 = data3['arr_0']
a3.shape

lab3 = read_csv("../csv/csv-3.csv")
lab3 = lab3.values[:, 1]

data4 = load('npz-4.npz')
lst4 = data4.files
a4 = data4['arr_0']
a4.shape

lab4 = read_csv("../csv/csv-4.csv")
lab4 = lab4.values[:, 1]

data5 = load('npz-5.npz')
lst5 = data5.files
a5 = data5['arr_0']
a5.shape

lab5 = read_csv("../csv/csv-5.csv")
lab5 = lab5.values[:, 1]

#############################################################################
###dividing train-test datasets (Randomly dividing the train-test into 4:1 ratio with equal proportion of 0:1 class)

## combing all files 
all_points = np.concatenate((a1,a2,a3,a4,a5), axis=0)
all_lab = np.concatenate((lab1,lab2,lab3,lab4, lab5), axis=0)

#extracting indices of datapoints of class '1' i(=394) and  of class '0' (=300)
z1 = np.where(all_lab==1)[0]
z0 = np.where(all_lab==0)[0]

##randomly selecting indices of 300 class '1' points (to keep equal positive:negative ratio) and dividing them into train:test in 4:1 ratio
sel = np.random.choice(z1, 300)
tr_posi = sel[0:240]
test_posi = sel[240:300] 

##randomly selecting indices of 300 class '0' points (to keep equal positive:negative ratio) and dividing them into train:test in 4:1 ratio
sel = np.random.choice(z0, 300)
tr_neg = sel[0:240]
test_neg = sel[240:300]

##combine the two classes into as X_train and X_test set
tr_ind = np.concatenate((tr_posi,tr_neg))
X_train = all_points[tr_ind,:,:,:]
X_train_lab = all_lab[tr_ind]

test_ind = np.concatenate((test_posi,test_neg))
X_test = all_points[test_ind,:,:,:]
X_test_lab = all_lab[test_ind]

##shuffle the points in the two datasets
s1 = np.random.choice(range(0,480), 480)
X_train = X_train[s1,:,:,:]
X_train_lab = X_train_lab[s1]

s2 = np.random.choice(range(0,120), 120)
X_test = X_test[s2,:,:,:]
X_test_lab = X_test_lab[s2]

##################################################################################
##plot the pixel distribution of train data
plt.hist(X_train.ravel(), bins=50, density=True)
plt.xlabel("Pixel values")
plt.ylabel("Relative frequency")
plt.title("Distribution of pixels")
plt.savefig('raw_data_pixel_dist.png')
plt.clf()

####################################################
##Data Normalization

#trainset
mean = np.mean(X_train, axis=(2,3), keepdims=True)
std = np.std(X_train, axis=(2,3), keepdims=True)
X_train = (X_train - mean) / std

#testset
mean = np.mean(X_test, axis=(2,3), keepdims=True)
std = np.std(X_test, axis=(2,3), keepdims=True)
X_test = (X_test - mean) / std


#plotting the normalized pixel distribution 
plt.hist(X_train.ravel(), bins=50, density=True)
plt.xlabel("Pixel values")
plt.ylabel("Relative frequency")
plt.title("Normalized Distribution of pixels")
plt.savefig('normalized_data_pixel_dist.png')
plt.clf()
#######################################################

##creating Dataset class and loading the data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid

class MyDataset(Dataset):
	def __init__(self, data, target, transform=None):
		self.data = torch.from_numpy(data).float()
		self.target = torch.from_numpy(target).long()
		self.transform = transform
	def __getitem__(self, index):
		x = self.data[index]
		y = self.target[index]
		if self.transform:
			x = self.transform(x)
		return x, y
	def __len__(self):
		return len(self.data)

train_dataset = MyDataset(X_train, X_train_lab)
test_dataset = MyDataset(X_test, X_test_lab)

##loading data for batch enumeration 
train_dl = DataLoader(train_dataset, batch_size=40, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=40, shuffle=False)

##################################################################################
#importing libraries for model building and evaluation

from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import Dropout2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_



##defining model
class CNN(Module):
    # define model elements
	def __init__(self, n_channels):
		super(CNN, self).__init__()
                # input to first hidden layer
		self.hid1 = Conv2d(n_channels, 40, (3,3))
		kaiming_uniform_(self.hid1.weight, nonlinearity='relu')
		#activation function
		self.act1 = ReLU()
		# first pooling layer
		self.pool1 = MaxPool2d((2,2), stride=(2,2))
		#dropping features
		self.dp1 = Dropout2d(p=0.1)
		# second hidden layer
		self.hid2 = Conv2d(40, 40, (3,3))
		kaiming_uniform_(self.hid2.weight, nonlinearity='relu')
		self.act2 = ReLU()
		# second pooling layer
		self.pool2 = MaxPool2d((2,2), stride=(2,2))
		self.dp2 = Dropout2d(p=0.1)
		# fully connected layer
		self.hid3 = Linear(30*30*40, 100)
		kaiming_uniform_(self.hid3.weight, nonlinearity='relu')
		self.act3 = ReLU()
		# output layer
		self.hid4 = Linear(100, 2)
		xavier_uniform_(self.hid4.weight)
		#to probabilities
		self.act4 = Softmax(dim=1)
	# forward propagate input
	def forward(self, X):
		# input to first hidden layer
		#print(X.shape)
		X = self.hid1(X)
		#print(X.shape)
		X = self.act1(X)
		#print(X.shape)
		X = self.pool1(X)
		#print(X.shape)
		X = self.dp1(X)
		#print(X.shape)
		# second hidden layer
		X = self.hid2(X)
		#print(X.shape)
		X = self.act2(X)
		#print(X.shape)
		X = self.pool2(X)
		#print(X.shape)
		X = self.dp2(X)
		#print(X.shape)
		# flatten
		X = X.view(-1, 30*30*40)
		#print(X.shape)
		# third hidden layer
		X = self.hid3(X)
		#print(X.shape)
		X = self.act3(X)
		#print(X.shape)
		# output layer
		X = self.hid4(X)
		#print(X.shape)
		X = self.act4(X)
		#print(X.shape)
		return X

## train the model
def train_model(train_dl, model):
	#define the optimization
	criterion = CrossEntropyLoss()
	optimizer = SGD(model.parameters(), lr=0.00003, momentum=0.9)
	losses = list()
	# enumerate epochs
	for epoch in range(20):
		print(epoch)
		# enumerate mini batches
		for i, (inputs, targets) in enumerate(train_dl):
			# clearing the gradients
			optimizer.zero_grad()
			# computing the model output
			mod_out = model(inputs)
			#print(mod_out)
			# calculate loss
			loss = criterion(mod_out, targets)
			loss_n = loss.item()
			losses.append(loss_n)
			# credit assignment
			loss.backward()
			# update model weights
			optimizer.step()
	return losses

## evaluate the model
def evaluate_model(test_dl, model):
	predictions, actuals, pred_score = list(), list(), list()
	for i, (inputs, targets) in enumerate(test_dl):
		# evaluate the model on the test set
		mod_out = model(inputs)
		# retrieve numpy array
		mod_out = mod_out.detach().numpy()
		actual = targets.numpy()
		print(mod_out.shape)
		##extracting scores 
		score = mod_out[:,1]
		score_l = score.tolist()
		pred_score = pred_score + score_l
		# convert to class labels
		mod_out = argmax(mod_out, axis=1)
		# reshape for stacking
		actual = actual.reshape((len(actual), 1))
		mod_out = mod_out.reshape((len(mod_out), 1))
		# store
		predictions.append(mod_out)
		actuals.append(actual)
	predictions, actuals = vstack(predictions), vstack(actuals)
	# calculate accuracy
	acc = accuracy_score(actuals, predictions)
	print('Accuracy achieved: %.2f' % acc)
	return pred_score

# define the CNN
model = CNN(1)
#train the model
loss_score = train_model(train_dl, model)
#evaluate the model
test_score = evaluate_model(test_dl, model)

#############################################################################################
##calculating model evaluation params

#import lib
from sklearn import metrics

##plotting learning cruve
plt.plot(loss_score)
plt.ylabel('loss')
plt.xlabel('Training Data points')
plt.title("Learning Curve")
plt.savefig('mod1_loss_curve.png')
plt.clf()

test_labels = X_test_lab.tolist()
fpr, tpr, _ = metrics.roc_curve(test_labels,  test_score)

#plotting ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC Curve")
plt.savefig('mod1_ROC.png')
plt.clf()

prec, recall, _ = metrics.precision_recall_curve(test_labels,  test_score)

#plotting prec-recall curve
plt.plot(recall, prec)
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title("PR Curve")
plt.savefig('mod1_PRC.png')
plt.clf()


##Area Under the Curve
roc_auc = metrics.auc(fpr, tpr)
prc_auc = metrics.auc(recall, prec)
print('ROC AUC: %.2f' % roc_auc)
print('PRC AUC: %.2f' % prc_auc)

