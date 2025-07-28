import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rmn import RMN
m = RMN()
# preceding * for extracting only keys and values ,removing unwanted characters
#image = m.video_demo()
cols = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
ln = len(cols)
path =  'https://github.com/aks2908/FaceEmotion/frm/'
vl = list()
ky = list()
lst = list()
for nframe in range(100):
   fig = path + 'frame_' + str(nframe)+ '.jpg'
   img = cv2.imread(fig,cv2.IMREAD_COLOR)
   results = m.detect_emotion_for_single_frame(img)
   kk = results[0]
   # Extracting all dictionary values
   res = [val for val in kk.values()]
   rr= res[6]
   vl = list()
   ky = list()
   for k in range(ln):
     ky.append(*rr[k].keys())
     vl.append(*rr[k].values())
   lst.append(vl)

df = pd.DataFrame(lst ,columns = cols)
df.to_csv('https://github.com/aks2908/FaceEmotion/data.csv')
