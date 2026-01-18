import pickle
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st


# Load the tokenizer
with open ("tokenizer.pickle","rb") as handle:
    tokenizer=pickle.load(handle)

#import the model
model=load_model ("LSTM_RNN_Hamlet_Model.h5")

def predict_next_word (model, tokenizer, text, max_sequence_len):
    tokenization_list = tokenizer.texts_to_sequences([text])[0]
    if len(tokenization_list) >= max_sequence_len:
        tokenization_list = tokenization_list [-(max_sequence_len-1):]
    padded_tokens = pad_sequences([tokenization_list],maxlen=max_sequence_len-1, padding = 'pre')
    prediction = model.predict(padded_tokens, verbose = 0)
    index_of_highest_probability = np.argmax(prediction, axis=1)
    for word,index in tokenizer.word_index.items():
        if index == index_of_highest_probability:
            return word
    return None

st.title("Next Word Prediction using LSTM RNN")

input_text = st.text_input("Enter the input text for the prediction:")

if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"The predicted next word is: {next_word}")
