# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 00:34:06 2023

@author: Sahil Chawla  Saurav Raj Kartik attri  Prof.Saurabh Rastogi
"""

import numpy as np
import pickle

import streamlit as st

loaded_model=pickle.load(open('D:/ckd/CKD_detection/trained_model.sav','rb')) 

#cretaing a function for prediction
def ckd_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)

    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]==1):
      return 'You are suffering from ckd'
    else:
      return 'You are not suffering from ckd' 
  
def main():
    #giving a title
    st.title('CKD Predictor')
    
    #taking input 
    white_blood_cell_count=st.text_input('White blood cells count')
    blood_urea=st.text_input('Blood Urea')
    blood_glucose_random=st.text_input('Blood Glucose')	
    serum_creatinine=st.text_input('Creatine Value')
    packed_cell_volume=st.text_input('packed Cell Volume')	
    albumin=st.text_input('Albumin Value')	
    haemoglobin=st.text_input('Haemoglobin')
    age=st.text_input('Age of Person')	
    sugar=st.text_input('sugar Value')	
    hypertension=st.text_input('hypertension')
    
    #code for prediction 
    diagnosis=''
    
    #creating a button forprediction
    
    if st.button('Predict'):
        diagnosis = ckd_prediction([white_blood_cell_count,blood_urea,blood_glucose_random,serum_creatinine,packed_cell_volume,albumin,haemoglobin,age,sugar,hypertension])
    
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()