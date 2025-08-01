import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg') # Or 'QtAgg', 'GTK3Agg', etc.
st.header('Display video')
path = "https://github.com/aks2908/FaceEmotion/blob/main/modi.mp4"
video_file = open(path,'rb')
video_bytes = video_file.read()
st.video(video_bytes)
st.header('Emotion Analytics of video')
#----------Plots-------------
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('emotion2.csv')
data = data.iloc[:,1:8]
ang = data.iloc[:,0]

dst = data.iloc[:,1]
fear = data.iloc[:,2]
Happy = data.iloc[:,3]
Sad = data.iloc[:,4]
Surp = data.iloc[:,5]
Neut = data.iloc[:,6]
st.header('Facial emotion result of video with only 10 rows displayed')
st.table(data.head(10))
chart_data = np.array(data)
[r ,c] = chart_data.shape
t = np.arange(r)
st.header('Bar plots of facial emotions from video')
ax1 = plt.subplot(2,3,1)
plt.bar(t,ang,label='Angry')
plt.xlabel('data points')
plt.ylabel('emotion level')
plt.legend()
ax2 = plt.subplot(2,3,2)
plt.bar(t,dst,label='Disgust')
plt.bar(t,fear,label='Fear')
plt.bar(t,Happy,label='Happy')
plt.xlabel('data points')
plt.legend()

ax3 = plt.subplot(2,3,3)
plt.bar(t,Sad,label='Sad')
plt.xlabel('data points')
plt.legend()
ax4=plt.subplot(2,3,4)
plt.bar(t,Surp,label='Surprise')
plt.xlabel('data points')
plt.ylabel('emotion level')
plt.legend()
ax5 = plt.subplot(2,3,5)

plt.bar(t,Neut,label='Neutral')
plt.xlabel('data points')
plt.legend()
st.pyplot(plt.gcf())
st.header('Line Plots of facial emotions from  video')
dat0 = data.iloc[:,0]
dat14 = data.iloc[:,1:4]
Sad = data.iloc[:,4]
Surp = data.iloc[:,5]
Neut = data.iloc[:,6]
st.title('Plot of Anger')
st.line_chart(dat0)
st.title('Plot of Disgust,Fear and Happiness' )
st.line_chart(dat14)
st.title('Plot of Sad emotion')
st.line_chart(Sad)
st.title('Plot of Surprise emotion ')
st.line_chart(Surp)
st.title('Plot of Neutral emotion')
st.line_chart(Neut)

st.write('-----------END------------------')
