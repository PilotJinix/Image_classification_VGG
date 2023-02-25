#Import library yang dibutuhkan
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.applications.mobilenet_v2 import MobileNetV2

import json
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

#Deklarasikan model yang akan dipakai
model = EfficientNetV2M()
#model = MobileNetV2()

#Tampilkan layer-layer model VGG16
model.summary()

response = open('clasification_asset.json')
klasifikasi = json.load(response)


def check_value(data, val):
    return any(check['No']==val for check in data)

upload = st.file_uploader('Masukkan foto dalan format PNG atau JPG', type=['png', 'jpg'])
if upload is not None:
    imgs = []
    plt.figure(figsize=(15, 6))
    r = Image.open(upload)
    img = np.array(r)
    plt.subplot(1, 1, 1)  # Menampilkan ke matplotlib
    img = cv2.resize(img, (480, 480))  # Resize sesuai input VGG 16
    imgs.append(img)  # Kumpulkan semua image yang telah di preproses ke imgs
    imgs = np.array(imgs)  # Konversi ke Np Array

    yh = model.predict(imgs)
    if (check_value(klasifikasi, np.argmax(yh))):
        st.success(f'Foto yang anda masukkan termasuk kategori : '+ klasifikasi[np.argmax(yh)]['Asset'].upper(), icon="✅")
    else:
        st.warning('Maaf Data Tidak Ditemukan', icon="⚠️")




