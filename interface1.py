import argparse
import numpy as np
import PIL
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import copy
import requests
from io import BytesIO
import math
import keyboard
import random
import os
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import streamlit as st

from sklearn.cluster import KMeans
import pickle
from joblib import dump, load

filename_model_claster = 'model_clasteris.joblib'
kmeans = load(filename_model_claster)
K = 1024

def ConvertImageToVector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    des = [d.pt for d in kp]
    prediction = kmeans.predict(des)
    vector, _ = np.histogram(prediction, bins=K)
    return vector

def main():
    kn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    data = pd.read_csv('data_full.csv')
    encoded_data = []
    for data_string in data['vector']:
        ds = data_string.replace("[", "").replace("]", "").replace("\n", " ")
        numpy_array = np.fromstring(ds, sep=' ', dtype=int)
        encoded_data += [numpy_array]
    kn_model.fit(encoded_data)

    st.title("Поиск похожих изображений")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if (uploaded_file):
        img = PIL.Image.open(uploaded_file).convert("RGB")
        img = PIL.ImageOps.exif_transpose(img)
        st.image(img)
        
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        input_vector = ConvertImageToVector(img)
        distances, indices = kn_model.kneighbors([input_vector])

        st.header('Похожие изображения:')
        image_paths = data['path_to_img']
        nearest_image_paths = [image_paths[i] for i in indices[0]]
        for path in nearest_image_paths:
            st.image(path)




if __name__ == "__main__":
    main()