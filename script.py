#!/usr/bin/env /home/priya/softwares/envs/pytorch_env/bin/python

##visualizing sample images from first file
import os
from numpy import load
from pandas import read_csv
import numpy as np

#setting the npz files folder as working dir
os.chdir('/home/priya/post_doc_assesement/Angio_Toy_Dataset/npz')

#loading img and lables
data1 = load('npz-1.npz')
lst = data1.files
print(lst)
a1 = data1['arr_0']
a1.shape

lab1 = read_csv("../csv/csv-1.csv")
lab1 = lab1.values[:, 1]

#finding indices of first 4 keyframes and 4 non-key frames
kf = np.where(lab1==1)
nkf = np.where(lab1==0)
ind = np.concatenate([kf[0][:4], nkf[0][:4]])

#plotting the 8 images in one frame 
from matplotlib import pyplot as plt
fig = plt.figure()
axes=[]

for i in range(len(ind)):
	axes.append(fig.add_subplot(2,4, i+1))
	#sub_title = ("image_"+str(ind[i])+"_"+ str(lab1[ind[i]]))
	if lab1[ind[i]]==1:
		sub_title = ("img_"+ str(ind[i]+1)+"_key")
	else:
		sub_title = ("img_"+str(ind[i]+1)+"_non_key")
	axes[-1].set_title(sub_title)
	plt.imshow(a1[ind[i],0,:,:], cmap='gray')

plt.tight_layout()
#plt.show()

##to save the image in png
plt.savefig('sample_images.png')





