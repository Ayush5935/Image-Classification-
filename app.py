import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import load_model
import  streamlit as st
import numpy as np
st.header('Image Classification Model')
model = load_model("model path")
data_cat = [ 'apple',
            'banana',
            'bell pepper',
            'cabbage',
            'capsicum'
]

img_height = 180
img_width = 180 

#image = " relative path image location in local"

image = st.text_input('Enter Image name','Apple.jpeg')

image_load = tf.keras.utils.load_img(image,target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
#image=image[31:len(image)-31]
#st.image(image)
st.image(image , width = 200)
#st.write('veg/fruit in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))

st.write('Veg/Fruit in image is' + data_cat[np.argmax(score)])
st.write('With accuracy' + str(np.max(score)*100))

#streamlit run app.py