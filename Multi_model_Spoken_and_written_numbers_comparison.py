#!/usr/bin/env python
# coding: utf-8

# Recognize whether an image of a hand- written digit and a recording of a spoken digit refer to the same or different number

# In[1]:


#Libraries Import
import numpy as np
from keras.models import Model
from keras import layers
from keras import Input
import matplotlib.pyplot as plt
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split

#For model Visulization
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

max_len_speak_frames=93
speak_frame_feature=13
img_height=img_width=28


# ### Data Preprocessing:

# In[2]:


# Data set Reading from Dataset Folder.
written_train=np.load('Datasets/written_train.npy',allow_pickle=True)
written_test=np.load('Datasets/written_test.npy',allow_pickle=True)
spoken_train=np.load('Datasets/spoken_train.npy',allow_pickle=True)
spoken_test=np.load('Datasets/spoken_test.npy',allow_pickle=True)
match_train0=np.load('Datasets/match_train.npy',allow_pickle=True)


# In[3]:


#category defining, As our match_train consists of False  adn true so we need to convert it to 0 and 1. If its false then 0.
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
match_train0=labelencoder_y.fit_transform(match_train0)


# Speak data consists of variable length, and is given as an array of shape (N, 13), where N is the number of frames in the recording, and 13 the number of MFCC features. First apply padding operation to make it same length sequence, so that vectorization allows code to efficiently perform the matrix operations on the batch. The pad_sequences() function in the Keras deep learning library can be used to pad variable length sequences.

# In[4]:


from keras.preprocessing.sequence import pad_sequences
# truncate sequence
speak_truncated_train= pad_sequences(spoken_train,maxlen=max_len_speak_frames, dtype='float')
speak_truncated_test= pad_sequences(spoken_test,maxlen=max_len_speak_frames, dtype='float')
print ('Pad Spoken data shape :',speak_truncated_train.shape)


# We are converting the image shape to img_height,img_width,1 so that we can use it in conv2 layer.

written_train0=written_train.reshape(written_train.shape[0],img_height,img_width,1)
written_test0=written_test.reshape(written_test.shape[0],img_height,img_width,1)


# ### Model Building:
# We choose multi model approach with lstm and Cnn based models used for speak and image respectively. And concatenated the both model output then apply binary cross entropy loss 

# In[5]:


# a single input layer 
input1 =Input(shape=(max_len_speak_frames, speak_frame_feature))
# x1 =layers.LSTM(40, activation="relu", dropout=0.25, recurrent_dropout=0.25)(input1)

x1 =layers.CuDNNLSTM(50)(input1)
x1=layers.BatchNormalization()(x1)
x1=layers.Activation('relu')(x1)
x1 =layers.Dropout(0.2)(x1)

x1 =layers.Dense(256)(x1)
x1=layers.BatchNormalization()(x1)
x1=layers.Activation('relu')(x1)
x1 =layers.Dropout(0.2)(x1)
x1 =layers.Dense(128, activation="relu")(x1)


input2 = Input(shape=(img_height,img_width,1))
x2 =layers.Conv2D(32, kernel_size=(3, 3))(input2)
x2=layers.BatchNormalization()(x2)
x2=layers.Activation('relu')(x2)
x2 =layers.Dropout(0.1)(x2)

x2 =layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x2)
x2=layers.BatchNormalization()(x2)
x2=layers.Activation('relu')(x2)
x2 =layers.MaxPooling2D(pool_size=(2, 2))(x2)

x2 =layers.Dropout(0.25)(x2)
x2 =layers.Flatten()(x2)
x2=layers.BatchNormalization()(x2)
x2 =layers.Dense(128, activation="relu")(x2)
x2 =layers.Dropout(0.5)(x2)


concatenated = layers.concatenate([x1, x2], axis=-1)

# output layer
predictions = Dense(1, activation='sigmoid')(concatenated)

# At model instantiation, we specify the two inputs and the output:
model = Model([input1, input2], predictions)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[6]:


## For saving model image but required Graphviz software.
# from keras.utils import plot_model
# plot_model(model, to_file='modelssss.png',show_layer_names=False)


# ### Model Training:

# In[7]:


#class_weight=class_weights,
val_acc=[]
acc=[]
loss=[]
val_loss=[]

for i in range(20):
    ## to solve class imbalance problem we choose random almost equal length of data. different data sample for every for loop iteration.
    new_index=np.unique(np.concatenate(((np.random.randint(0,45000,5000).astype('int')),np.where(match_train0>0)[0].astype('int'))))
    
    # new data sample train test spliting
    spoken_train,spoken_test,written_train,written_test,match_train,match_test=train_test_split(speak_truncated_train[new_index],written_train0[new_index],match_train0[new_index],test_size=0.2,random_state=0)
    hist=model.fit([spoken_train,written_train], match_train, epochs=20,batch_size=1024, validation_data=([spoken_test,written_test],match_test))
    
    # accuracy and loss saving for all epochs.
    acc=acc+hist.history['acc']
    val_acc=val_acc+hist.history['val_acc']
    loss=loss+hist.history['loss']
    val_loss=val_loss+hist.history['val_loss']


# ### Model Evaluation

# In[19]:


plt.figure(figsize=[10,8])
plt.plot(acc,label='Train Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.title('Train/Validation Accuracy comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Train_Validation_Accuracy_comparison.png')
plt.show()


# In[20]:


plt.figure(figsize=[10,8])
plt.plot(loss,label='Train Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title('Train/Validation Loss comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Train_Validation_Loss_comparison.png')
plt.show()


# In[10]:


print ('Score on last training data [loss,acc]: ',model.evaluate([speak_truncated_train,written_train0], match_train0))
##Confusion matrix prediction on last whole training Dataset.
predicted=model.predict([speak_truncated_train,written_train0])

predicted[predicted>0.5]=1
predicted[predicted<0.51]=0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(match_train0, predicted)
print ('Confusion Matrix on all training data: ',cm)


# In[11]:


print ('Score on last test data [loss,acc]: ', model.evaluate([spoken_test,written_test], match_test))

##Confusion matrix prediction on last test Dataset.
predicted=model.predict([spoken_test,written_test])
predicted[predicted>0.5]=1
predicted[predicted<0.51]=0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(match_test, predicted)
print ('Confusion Matrix on last test data: ',cm)


# In[14]:


import pandas as pd


# In[16]:


pd.DataFrame(cm)


# In[13]:


858+19


# In[12]:


print ('Prediction on test data and saving output prediction as bolean values in result.npy file.')
test_predicted=model.predict([speak_truncated_test,written_test0])

test_predicted[test_predicted>0.5]=True
test_predicted[test_predicted<0.51]=False

test_predicted=(test_predicted.astype('int')>0).reshape(-1,)

np.save('result.npy',test_predicted,allow_pickle=True)


# In[ ]:




