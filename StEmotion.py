import streamlit as st
import pandas as pd
import numpy as np

st.title('Emotion detection from face image')

import cv2
from rmn import RMN
m = RMN()
m.video_demo()
#image = cv2.imread("some-image-path.png")
image = cv2.imread("")
results = m.detect_emotion_for_single_frame(image)

image = m.draw(image, results)
cv2.imwrite("output.png", image)
cv2.imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()

