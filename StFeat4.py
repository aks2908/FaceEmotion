import streamlit as st
import pandas as pd
import numpy as np
from feat import Detector
import os
#from feat.utils.io import get_test_data_path
from feat.utils.io import read_feat
from feat.plotting import imshow
import scipy
import matplotlib.pyplot as plt
# initialize Detector
detector = Detector(emotion_model="resmasknet")

st.title("Facial Emotion  Processing App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None :
   file_name = uploaded_file.name
   test_dir = os.path.dirname(os.path.realpath(__file__))
   imgpath = os.path.join(test_dir, file_name)
   st.image(imgpath)
   single_face_prediction = detector.detect_image(imgpath)
   st.write('single_face_prediction.emotions')
   st.write(single_face_prediction.emotions)
   lst = single_face_prediction.emotions
   dt = np.array(lst)
   dfrm = pd.DataFrame(lst)
   cols = dfrm.columns
   yy = dfrm.iloc[0, :]
   yy = np.float32(yy)
   xx = cols
   plt.title('Facial Emotions plot')
   plt.bar(cols, yy, color='skyblue')

   # Add labels and title
   plt.xlabel('Emotion types')
   plt.ylabel('Values')

st.pyplot(plt.gcf())
st.write('------------END--------------')