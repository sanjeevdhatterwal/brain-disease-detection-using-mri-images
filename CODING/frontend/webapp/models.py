
from django.db import models
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json
from PIL import Image


from tensorflow.keras.models import load_model
cnn=load_model(r"C:\Users\sanje\Music\mental-illness\CODING\frontend\mental_cnn.h5")
vgg16=load_model(r"C:\Users\sanje\Music\mental-illness\CODING\frontend\mental_vgg16.h5")
# inv3 =load_model(r'C:\Users\ST-0006\Documents\dont touch\ANAMOLY DETECTION ON ASTRONOMICAL OBJECTS\Code\front end\ASTRO_inv3.h5')
#svm =load_model(r'D:\Deep_learning_Deployment\Plant_seedling_classification\Front End Code\plant_seedling_vgg16.h5')



def predict(img,algo): 
	file = Image.open(img)
	img = file.resize((48,48))
	img_array = np.asarray(img).astype(np.uint8)
	res = img_array.reshape([-1,48, 48,3])
	print(res.shape)
	if algo=='vgg16':
		y_pred=np.argmax(vgg16.predict(res),axis=1)
		return y_pred[0]
	else:
		y_pred=np.argmax(cnn.predict(res),axis=1)
		return y_pred[0]



