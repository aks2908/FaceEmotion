import streamlit as st
from PIL import Image
import numpy as np

st.title("Image Processing App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Image Processing Options")

    # Example: Convert to grayscale
    if st.button("Convert to Grayscale"):
        # Convert PIL Image to NumPy array for processing
        img_array = np.array(original_image)

        # Convert to grayscale (simple average of R, G, B channels)
        # For more sophisticated grayscale conversion, consider OpenCV or specific color conversion formulas.
        grayscale_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        grayscale_image = Image.fromarray(grayscale_array.astype(np.uint8))

        st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

    # You can add more image processing functionalities here,
    # such as resizing, cropping, applying filters (blur, sharpen), etc.,
    # using libraries like PIL (Pillow) or OpenCV.