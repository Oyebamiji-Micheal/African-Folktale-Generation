import streamlit as st
import numpy as np
import joblib

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Header
st.write("<h3 align='center'>African Folktale Generation</h3>", unsafe_allow_html=True)

# st.image("images/repo-cover.jpg")

st.write("""### Model Inference""")

# Input text
input_text = st.text_area(
    "Enter an example text:",
    placeholder="Tortoise and the hare"
)


tokenizer_path = "models/tokenizer.joblib"
model_path = "models/model.h5"
seq_length = 100


def load_tokenizer(tokenizer_path):
    tokenizer = joblib.load(tokenizer_path)
    return tokenizer


def load_model(model_path):
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


# Button and spinner
if st.button("Generate Story"):
    if input_text:
        with st.spinner("Generating, please wait..."):
            generated_text = generate_text_base_word(tokenizer_path, model_path, input_text, 300)
        st.success("Generation complete!")
        st.write("##### == Generated Text ====")
        st.write(generated_text)
    else:
        st.warning("Please enter text to translate.")
