from transformers import TFAutoModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from preprocess import *
from tensorflow.data import Dataset
from config import *
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(
  page_title='Vietnamese ABSA', 
  page_icon='https://cdn-icons-png.flaticon.com/512/8090/8090669.png', 
  layout="centered")
st.image(image=Image.open('assets/homepage.png'), caption='Aspect-based Sentiment Analysis')

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
def tokenize_function(dataset):
    clean_texts = list(map(text_preprocess, dataset['comment']))
    return tokenizer(clean_texts, max_length=tokenizer.model_max_length, padding='max_length', truncation=True)

# Prepare model
from transformers import TFAutoModel
from tensorflow import keras as K

def create_model(optimizer):
    # https://riccardo-cantini.netlify.app/post/bert_text_classification
    inputs = {
        'input_ids'     : K.layers.Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_ids'),
        'token_type_ids': K.layers.Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='token_type_ids'),
        'attention_mask': K.layers.Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='attention_mask'),
    }
    pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
    hidden_states = pretrained_bert(inputs).hidden_states

    # https://github.com/huggingface/transformers/issues/1328
    pooled_output = K.layers.concatenate(
        tuple([hidden_states[i] for i in range(-4, 0)]),
        name = 'last_4_hidden_states',
        axis = -1
    )[:, 0,: ]
    x = K.layers.Dropout(0.2)(pooled_output)
    print(pooled_output)

    outputs = K.layers.concatenate([
        K.layers.Dense(
            units = 4,
            activation = 'softmax',
            name = label.replace('#', '-').replace('&', '_'),
        )(x) for label in ASPECTS
    ], axis = -1)

    model = K.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy')
    return model

model = create_model(None)
model.load_weights(f'multitask_approach_phobert.h5')
model.summary()

# predict
def predict(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1)
    
def tokenize_function(dataset):
    clean_texts = list(map(text_preprocess, dataset['comment']))
    return tokenizer(clean_texts, max_length=tokenizer.model_max_length, padding='max_length', truncation=True)
    
def preprocess_tokenized_dataset(tokenized_dataset, tokenizer, batch_size = 10, shuffle=False):
    # tf_dataset = to_tensorflow_format(tokenized_dataset)
    # features = {x: tf_dataset[x] for x in tokenizer.model_input_names}
    
    features = dict(tokenized_dataset)

    tf_dataset = Dataset.from_tensor_slices((features))
    if shuffle: tf_dataset = tf_dataset.shuffle(buffer_size=len(tf_dataset))
    return tf_dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Driver
sentence = st.text_input(label='')

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if st.button('Analyze'):
  if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        tokenized_data = tokenize_function(data)
        
        test_tf_dataset = preprocess_tokenized_dataset(tokenized_data,  tokenizer)
        y_pred = predict(model, test_tf_dataset, 10, verbose=1)
        print(y_pred)
        
        #y_pred = [[0,1,0,1,2,3,1,1,2,2,2],[0,1,0,1,2,0,1,1,2,2,2]]
      
        aspect_polarity_dict = {aspect: [] for aspect in ASPECTS}
        print(aspect_polarity_dict)
        for sample_id in range(len(y_pred)):
            for aspect_id, aspect in enumerate(ASPECTS):
                if y_pred[sample_id][aspect_id] != 0:
                    aspect_polarity_dict[aspect].append(REPLACEMENTS[y_pred[sample_id][aspect_id]])
        print(aspect_polarity_dict)
        
        # Tạo biểu đồ tròn cho Aspect
        aspect_counts = {aspect: len(aspect_polarity_dict[aspect] ) for aspect in aspect_polarity_dict}
        print(aspect_counts)
        plt.figure(figsize=(6, 6))
        plt.pie(aspect_counts.values(), labels=aspect_counts.keys(), autopct='%1.1f%%')
        plt.title('Aspect Distribution')
        plt.axis('equal')
        aspect_chart = plt.gcf()

        # Tạo biểu đồ cho 10 aspect
        pol_pie_charts = {}
        for aspect in ASPECTS:
            plt.figure(figsize=(6, 6))
            counter = Counter(aspect_polarity_dict[aspect])
            plt.pie(counter.values(), labels=counter.keys(), autopct='%1.1f%%')
            plt.title(f'Polarity Distribution Of {aspect}')
            plt.axis('equal')
            pol_pie_charts[aspect] = plt.gcf()

        # Plot aspect chart
        st.write('*Aspect Distribution:*')
        st.pyplot(aspect_chart)

        # Plot 10 polarity pie chart for each aspect
        for aspect in ASPECTS:
            st.write(f'Polarity Distribution of {aspect}')
            st.pyplot(pol_pie_charts[aspect])
  else: 
    clean_sentence = text_preprocess(sentence)
    tokenized_input = tokenizer(clean_sentence, padding='max_length', truncation=True)
    features = {x: [[tokenized_input[x]]] for x in tokenizer.model_input_names}
    pred = predict(model, Dataset.from_tensor_slices(features))
    
    sentiments = map(lambda x: REPLACEMENTS[x], pred[0])
    d = []
    for aspect, sentiment in zip(ASPECTS, sentiments): 
        if sentiment: d.append(f'{aspect}#{sentiment}')
    st.markdown('*')
    st.write('*Sentence:*', sentence)
    st.write(', '.join(d))