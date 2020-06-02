#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
#    Jun 01, 2020 07:47:25 PM IST  platform: Windows NT

import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

from PIL import Image, ImageTk

import face_mask_support
import os.path
import cv2,glob
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = tk.Tk()
    top = Toplevel1 (root)
    face_mask_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    face_mask_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:

    def detect_mask(self):

        data_path='data'
        category_dir = os.listdir(data_path) #it  lists outs the dirs in data
#print(category_dir)
        labels_dir = [i for i in range(len(category_dir))] # I have labeled the dirs as "0" and "1"
#print(labels_dir)
        labels_dict = dict(zip(category_dir, labels_dir)) # creating a dictionay of cat and lab
        #print(labels_dict)

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
                #print(result)
                label=np.argmax(result,axis=1)[0]
                #print(label)
      
                cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
            cv2.imshow("LIVE",img)
            key=cv2.waitKey(1)
    
            if(key==ord("q")):
                break
        
        cv2.destroyAllWindows()
        source.release()







    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font9 = "-family {Castellar} -size 23 -weight bold -underline "  \
            "1"

        top.geometry("600x577+650+150")
        top.minsize(148, 1)
        top.maxsize(1924, 1055)
        top.resizable(1, 1)
        top.title("New Toplevel")
        top.configure(background="#000000")

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=-0.017, rely=-0.017, height=468, width=616)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(cursor="fleur")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(foreground="#000000")
        photo_location = os.path.join(prog_location,"rsz_1mask.jpg")
        global _img0
        _img0 = ImageTk.PhotoImage(file=photo_location)
        self.Label1.configure(image=_img0)

        self.Button1 = tk.Button(top)
        self.Button1.place(relx=0.0, rely=0.78, height=133, width=606)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#00ff40")
        self.Button1.configure(borderwidth="15")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(font=font9)
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Check for Mask!''')
        self.Button1.configure(command = self.detect_mask)

if __name__ == '__main__':
    vp_start_gui()





