import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("Handwritten Digit Image Generator")

@st.cache_resource
def load_gan():
    return load_model("mnist_gan_generator.h5")

generator = load_gan()
latent_dim = 100

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))
if st.button("Generate Images"):
    cols = st.columns(5)
    for i in range(5):
        noise = np.random.normal(0, 1, (1, latent_dim))
        label = np.array([[digit]])
        gen_img = generator.predict([noise, label], verbose=0)
        gen_img = 0.5 * gen_img + 0.5  # Rescale to [0,1]
        gen_img = gen_img[0, :, :, 0]
        cols[i].image(gen_img, caption=f"Sample {i+1}", width=100)
