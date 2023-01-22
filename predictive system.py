# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
 
#loading the saved model
loaded_model=pickle.load(open('D:/ckd/CKD_detection/trained_model.sav','rb')) 

input_data=(7800.0,121.0,36.0,1.2,44.0,1.0,15.4,48.0,0.0,1)

input_data_as_numpy_array=np.asarray(input_data)

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==1):
  print('The person is suffering from ckd')
else:
  print('The person is not suffering from ckd') 