import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle

model = load_model('model.h5')
with open('Tokenizer.pickle','rb') as file:
    tokenizer = pickle.load(file)


def predict_with_text(model,text):
  tokenized_text = tokenizer.texts_to_sequences([text])[0]
  max_sequence_len = model.input_shape[1] + 1
  if len(tokenized_text) > max_sequence_len:
    tokenized_text = tokenized_text[-(max_sequence_len-1):]
  tokenized_text = pad_sequences([tokenized_text],maxlen=max_sequence_len-1,padding='pre')
  prediction = model.predict(tokenized_text,verbose=0)
  most_probable = np.argmax(prediction,axis=1)
  for word,index in tokenizer.word_index.items():
    if index == most_probable:
      return word
  return None



st.title("Next Word Predicition using LSTM")
input_text = st.text_input("Enter the sequence of words:")
if st.button("Predict next word!"):
    next_word = predict_with_text(model,input_text)
    st.write(f'Next predicted word : {next_word}')