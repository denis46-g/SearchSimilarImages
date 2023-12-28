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

import tkinter as tk
from tkinter import filedialog

from sklearn.cluster import KMeans
import pickle
from joblib import dump, load

def main():
    kn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    data = pd.read_csv('data_full2.csv')
    encoded_data = []
    for data_string in data['vector']:
        string_values = data_string.strip('[]\n').split()
        float_values = np.array([float(val) for val in string_values])
        encoded_data += [float_values]
    kn_model.fit(encoded_data)

    st.title("Поиск похожих изображений")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if (uploaded_file):
        img = PIL.Image.open(uploaded_file).convert("RGB")
        img = PIL.ImageOps.exif_transpose(img)
        st.image(img)
        cur_path = './VOCdevkit/VOC2012/JPEGImages/' + uploaded_file.name

        input_vector = data.loc[data['path_to_img'] == cur_path, 'index']
        input_vector = data['vector'].values[input_vector][0]
        string_values = input_vector.strip('[]\n').split()
        input_vector = np.array([float(val) for val in string_values])

        distances, indices = kn_model.kneighbors([input_vector])

        st.header('Похожие изображения:')
        image_paths = data['path_to_img']
        nearest_image_paths = [image_paths[i] for i in indices[0]]
        for path in nearest_image_paths:
            st.image(path)

if __name__ == "__main__":
    main()