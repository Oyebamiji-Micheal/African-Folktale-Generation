import streamlit as st
import numpy as np
import joblib
import time
from diffusers import StableDiffusionPipeline
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Header
st.write("<h3 align='center'>African Folktale Generation</h3>", unsafe_allow_html=True)

st.write("""### Model Inference""")

# Input text
input_text = st.text_area(
    "Enter an example text:",
    placeholder="Tortoise and the hare"
)

tokenizer_path = "models/tokenizer.joblib"
model_path = "models/model-initial.h5"
seq_length = 100

# Creating pipeline for Stable Diffusion
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float32  # Use float32 for CPU
)

def load_tokenizer(tokenizer_path):
    tokenizer = joblib.load(tokenizer_path)
    return tokenizer

def load_saved_model(model_path):
    model = load_model(model_path)
    return model

def generate_text_base_word(tokenizer_path, model_path, seed_text, num_words):
    tokenizer, model = load_tokenizer(tokenizer_path), load_model(model_path)
    result = []
    in_text = seed_text
    for _ in range(num_words):
        # Encode the input text
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        
        # Predict the next word
        yhat = model.predict(encoded, verbose=0)
        yhat = np.argmax(yhat, axis=1)
        
        # Map index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        
        # Append the word to the result
        in_text += ' ' + out_word
        result.append(out_word)
    
    return ' '.join(result)

# Function to generate a single image
def generate_image(prompt):
    images = pipeline(prompt).images  # Generate images based on the prompt
    return images[0]  # Return the first image

# Button and spinner
if st.button("Generate Story"):
    if input_text:
        with st.spinner("Generating, please wait..."):
            # Generate the text/story
            generated_text = generate_text_base_word(tokenizer_path, model_path, input_text, 200)
        
        st.success("Generation complete!")
        st.write("###### -------- Generated Text --------")
        st.write(generated_text)
        
        # Generate the image based on the story
        image_prompt = "A scene from the folktale: " + generated_text[:50]  # Use part of the generated text as a prompt
        with st.spinner("Generating image..."):
            image = generate_image(image_prompt)
        
        # Display the generated image
        st.image(image, caption="Generated Image from the Story", use_column_width=True)
    else:
        st.warning("Please enter story to generate.")
