import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from skimage import filters, feature, color

def load_image(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

def display_image(image, title="Image"):
    if image is None:
        st.write("No image to display.")
        return

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

def apply_filter(image, filter_type, sigma=1):
    if image is None:
        st.write("No image uploaded.")
        return None

    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    if filter_type == 'Mean':
        return filters.rank.mean(image, np.ones((5, 5)))
    elif filter_type == 'Gaussian':
        return filters.gaussian(image, sigma=sigma)
    elif filter_type == 'Canny':
        return feature.canny(gray, sigma=sigma)
    else:
        return image

st.title("Image Processing App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = load_image(uploaded_file)
    st.subheader("Original Image")
    display_image(original_image)

    filter_type = st.selectbox('Select Filter:', ['Mean', 'Gaussian', 'Canny'])
    sigma = st.slider('Sigma:', 0.1, 10.0, 1.0, 0.1)

    if st.button("Apply Filter"):
        processed_image = apply_filter(original_image, filter_type, sigma)
        st.subheader(f"{filter_type} Filtered Image")
        display_image(processed_image)

        if processed_image is not None:
            # Convert the processed image to bytes
            img = Image.fromarray((processed_image * 255).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            btn = st.download_button(
                label="Download Filtered Image",
                data=byte_im,
                file_name="filtered_image.png",
                mime="image/png"
            )
