import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('my_cnn_model.h5')

def process_image(img):
    img=img.resize((170,170)) #boyutunu 170 x 170 pixel yaptik
    img=np.array(img)
    img=img/255.0 #normalize ettik
    img=np.expand_dims(img,axis=0)
    return img

st.title('kanser resmi sınıflandırma')
st.write('Upload an image')

file=st.file_uploader('resmi sec',type=['jpeg','jpg','png'])

if file is not None:
    img=Image.open(file)
    st.image(img,caption='yklnene resim')
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)

    class_names=['kanserDeğil','Kanser']
    st.write(class_names[predicted_class])
