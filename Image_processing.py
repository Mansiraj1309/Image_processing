import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import sobel
from sklearn.cluster import KMeans
import seaborn as sns

def grayscale_transformation(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def histogram_equalization(image):
    gray = grayscale_transformation(image)
    return cv2.equalizeHist(gray)

def fourier_transform(image):
    gray = grayscale_transformation(image)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Avoid log(0) error
    return (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())  # Normalize

def edge_and_corner_detection(image):
    gray = grayscale_transformation(image)
    edges = cv2.Canny(gray, 100, 200)
    corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    return edges, corners

def kmeans_segmentation(image, k=3):
    reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reshaped)
    segmented = labels.reshape(image.shape[:2])
    colored_segmented = sns.color_palette("tab10", k)
    segmented_rgb = np.array([colored_segmented[label] for label in segmented.flatten()]).reshape(image.shape)
    return np.uint8(segmented_rgb * 255)

# Streamlit UI
st.title("Image Processing Tasks")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_container_width=True, channels="BGR")
    
    task = st.selectbox("Choose a task", ["Grayscale & Histogram Equalization", "Fourier Transform", "Edge & Corner Detection", "K-Means Segmentation"])
    
    if task == "Grayscale & Histogram Equalization":
        grayscale = grayscale_transformation(image)
        hist_eq = histogram_equalization(image)
        st.image([grayscale, hist_eq], caption=["Grayscale Image", "Histogram Equalized Image"], use_container_width=True, channels="GRAY")
    
    elif task == "Fourier Transform":
        magnitude_spectrum = fourier_transform(image)
        fig, ax = plt.subplots()
        ax.imshow(magnitude_spectrum, cmap='gray')
        ax.set_title("Fourier Transform Spectrum")
        ax.axis("off")
        st.pyplot(fig)
    
    elif task == "Edge & Corner Detection":
        edges, corners = edge_and_corner_detection(image)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(edges, cmap='gray')
        ax[0].set_title("Canny Edge Detection")
        ax[1].imshow(corners, cmap='gray')
        ax[1].set_title("Harris Corner Detection")
        st.pyplot(fig)
    
    elif task == "K-Means Segmentation":
        k = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
        segmented = kmeans_segmentation(image, k)
        st.image(segmented, caption="Segmented Image", use_container_width=True)
