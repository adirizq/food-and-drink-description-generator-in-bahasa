import re
import pickle
import zipfile
import os
import streamlit as st
import numpy as np
import pandas as pd
import urllib.request
import zipfile

from keras.models import load_model
from keras.utils import pad_sequences
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, set_seed

st.set_page_config(page_title="Food and Drink Description Generator in Bahasa", page_icon="🍽️", layout="centered")


@st.cache(allow_output_mutation=True, show_spinner=False, ttl=3600, max_entries=10)
def inference_init():
    lstm_path = 'models/lstm'
    gpt2_path = 'models/finetuned-gpt2'

    if not os.path.exists(lstm_path) and not os.path.exists(gpt2_path):
        with st.spinner("Downloading models... this may take awhile! \n Don't stop it!"):
            url = 'https://github.com/adirizq/data/releases/download/food_drink_desc_generator/models.zip'
            filename = url.split('/')[-1]

            urllib.request.urlretrieve(url, filename)

            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('models/')

    with st.spinner("Loading models... this may take awhile! \n Don't stop it!"):
        lstm_model = load_model(lstm_path)
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('cahya/gpt2-small-indonesian-522M')
        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_path, pad_token_id=gpt2_tokenizer.eos_token_id)

        with open('datasets/word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)

        with open('datasets/idx_to_word.pkl', 'rb') as f:
            idx_to_word = pickle.load(f)

    return lstm_model, gpt2_tokenizer, gpt2_model, word_to_idx, idx_to_word


lstm_model, gpt2_tokenizer, gpt2_model, word_to_idx, idx_to_word = inference_init()
set_seed(99)


def clean(data):
    cleaned_data = []
    unique_words = []

    for description in data:
        description = re.sub('[^a-zA-Z]', ' ', description)
        description = re.sub(r'\b\w{0,1}\b', ' ', description)
        description = description.lower().strip()
        description = description.split()
        cleaned_data.append(description)
        unique = list(set(description))
        unique_words.extend(unique)

    unique_words = set(unique_words)

    return cleaned_data, unique_words, len(unique_words)


def prepare_corpus(data, word_to_idx):
    sequences = []
    for line in data:
        tokens = line
        for i in range(1, len(tokens)):
            i_gram_sequence = tokens[:i+1]
            i_gram_sequence_ids = []

            for j, token in enumerate(i_gram_sequence):
                try:
                    i_gram_sequence_ids.append(word_to_idx[token])
                except:
                    i_gram_sequence_ids.append(0)

            sequences.append(i_gram_sequence_ids)
    return sequences


def lstm_generate_text(input_text, len_words):
    for _ in range(len_words):
        cleaned_data = clean([input_text])
        sequences = prepare_corpus(cleaned_data[0], word_to_idx)
        sequences = pad_sequences([sequences[-1]], maxlen=40-1, padding='pre')
        predict_x = lstm_model.predict(sequences, verbose=0)
        classes_x = np.argmax(predict_x, axis=1)
        output_word = ''
        output_word = idx_to_word[classes_x[0]]
        input_text = input_text + " " + output_word

    return input_text.title()


def gpt2_generate_text(input_text, len_words):
    encoded_input = gpt2_tokenizer.encode(input_text, return_tensors='pt')

    beam_outputs = gpt2_model.generate(
        encoded_input,
        max_length=len_words + 8,
        num_beams=5,
        num_return_sequences=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    value = []
    for i, beam_output in enumerate(beam_outputs):
        value.append(gpt2_tokenizer.decode(beam_output, skip_special_tokens=True))

    return value


st.title('🍽️ Food and Drink Description Generator in Bahasa')

with st.expander('📋 About this app', expanded=True):
    st.markdown("""
    * Food and Drink Description Generator in Bahasa app is an easy-to-use tool that allows you to generate food and drink description text in bahasa from starting text input and max words input.
    * There are two models you can generate description text from, they are LSTM and Fine-tuned GPT2.
    * Made by [Rizky Adi](https://www.linkedin.com/in/rizky-adi-7b008920b/).
    """)
    st.markdown(' ')

with st.expander('🧠 About text generation model', expanded=False):
    st.markdown("""
    #### Food and Drink Description Generator in Bahasa
    * Model are trained based on [Indonesia food delivery Gofood product list](https://www.kaggle.com/datasets/ariqsyahalam/indonesia-food-delivery-gofood-product-list) datasets by Reyhan Ariq Syahalam on Kaggle.
    * Model trained using [LSTM](https://keras.io/api/layers/recurrent_layers/lstm/) by Keras API and Fine-tuned [Pretrained GPT2](https://huggingface.co/cahya/gpt2-small-indonesian-522M) by Cahya Wirawan on HuggingFace.
    * **[Source Code](https://github.com/adirizq/indonesian-news-title-category-classifier)**
    """)
    st.markdown(' ')

st.markdown(' ')
st.markdown(' ')

st.header('📝 Food and Drink Description Generator')

with st.form("form"):
    input_text = st.text_input('Starting text', placeholder='Enter your description starting text here')
    len_words = st.number_input('Number of words', min_value=1, max_value=50, value=20)

    submitted = st.form_submit_button("Submit")

st.markdown(' ')

if submitted:
    with st.spinner('Generating description...'):
        result_lstm = lstm_generate_text(input_text, len_words)
        result_gpt2 = gpt2_generate_text(input_text, len_words)

    st.write('#### Result LSTM Model:')
    st.success(result_lstm)

    st.markdown(' ')

    st.write('#### Result Fine-tuned GPT2 Model:')
    for result in result_gpt2:
        st.success(result)
