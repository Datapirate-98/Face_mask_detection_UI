#!/usr/bin/env python
# coding: utf-8

# In[40]:


import cv2,glob
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop


# In[41]:


data_path='data'
category_dir = os.listdir(data_path) #it  lists outs the dirs in data
#print(category_dir)
labels_dir = [i for i in range(len(category_dir))] # I have labeled the dirs as "0" and "1"
#print(labels_dir)
labels_dict = dict(zip(category_dir, labels_dir)) # creating a dictionay of cat and lab
print(labels_dict)

data = []
target = []

for cat in category_dir:
    cat_path = os.path.join(data_path, cat) #it returns the category path in cat folder
    #print(cat_path)
    img_names = os.listdir(cat_path)
    #print(img_names)
    
    for name in img_names:
        img_path = os.path.join(cat_path, name)
        #print(img_path)
        img = cv2.imread(img_path)
        
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(grey, (128, 128))
        data.append(resize)
        target.append(labels_dict[cat])
        
data = np.array(data)/255.0
data = np.reshape(data, (data.shape[0], 128, 128, 1))
#print(data)
target = np.array(target)
target = np_utils.to_categorical(target)

np.save("data", data)
np.save("target", target)


# In[42]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.70): # I kept it 70% acc because i got best result in it u can try experimenting
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True


# In[43]:


data = np.load("data.npy")
target = np.load("target.npy")

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3,3), activation = "relu", input_shape = data.shape[1:]),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(2, activation = "softmax")
])

model.summary()

model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size = 0.1)


history=model.fit(
  train_data,
  train_target,
  epochs=20,
  callbacks=[callbacks],
  validation_split=0.2)

model.save("masked.h5")

"""test_loss, test_acc = model.evaluate(test_data, test_target)
print("\n Accuracy :   ", test_acc)
print("\n loss :   ", test_loss)
predictions = model.predit(test_data)
print(predictions)"""


# In[37]:


model = tf.keras.models.load_model("masked.h5")

classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

source=cv2.VideoCapture(0)

labels_dict={0:"without_mask",1:"with_mask"}
color_dict={0:(0,0,255),1:(0,255,0)}



while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(128,128))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,128,128,1))
        result=model.predict(reshaped)
        print(result)
        label=np.argmax(result,axis=1)[0]
        print(label)
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow("LIVE",img)
    key=cv2.waitKey(1)
    
    if(key==ord("q")):
        break
        
cv2.destroyAllWindows()
source.release()

